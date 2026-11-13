import re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)


root = Path(__file__).resolve().parents[1]
output = root / "results" / "word_embeddings"
output.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(root / "data-bin" / "group_sample_seed43.csv")
df = df[df["Score"].isin([1, 2, 4, 5])]
df = df.dropna(subset=["Text"]).copy()

df["label"] = (df["Score"] >= 4).astype(int)

df["Text"] = df["Text"].str.lower()
df["Text"] = df["Text"].str.replace(r"<[^>]+>", " ", regex=True)
df["Text"] = df["Text"].str.replace(r"\s+", " ", regex=True).str.strip()
df = df[df["Text"] != ""]

label_counts = df.groupby("Text")["label"].nunique()
df = df[df["Text"].isin(label_counts[label_counts == 1].index)]
df = df.drop_duplicates("Text")

embeddings = {}

with open(root / "data-bin" / "glove.6B.100d.txt", encoding="utf-8") as file:
    for line in file:
        parts = line.split()
        embeddings[parts[0]] = np.array(parts[1:], dtype=np.float32)


def review_vector(text):
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    vectors = [embeddings[word] for word in words if word in embeddings]

    if len(vectors) == 0:
        return np.zeros(100)

    return np.mean(vectors, axis=0)


train, test = train_test_split(
    df,
    test_size=0.20,
    random_state=43,
    stratify=df["label"],
)

X_train = np.array([review_vector(text) for text in train["Text"]])
X_test = np.array([review_vector(text) for text in test["Text"]])

y_train = train["label"]
y_test = test["label"]

model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

majority_label = y_train.mode()[0]
baseline = np.full(len(y_test), majority_label)


results = []


def show_results(name, predicted):
    scores = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predicted),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "Recall": recall_score(y_test, predicted, zero_division=0),
        "F1": f1_score(y_test, predicted, zero_division=0),
    }
    results.append(scores)
    print("\n" + name)
    for metric in ["Accuracy", "Precision", "Recall", "F1"]:
        print(metric + ":", round(scores[metric], 4))


print("Training reviews:", len(train))
print("Test reviews:", len(test))
print("Features per review:", X_train.shape[1])

show_results("Majority baseline", baseline)
show_results("Word embeddings + Gaussian Naive Bayes", predictions)
pd.DataFrame(results).to_csv(output / "metrics.csv", index=False)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=[0, 1],
    display_labels=["Negative", "Positive"],
    cmap="Blues",
)
plt.tight_layout()
plt.savefig(output / "confusion_matrix.png", dpi=200)
plt.close()

word_counts = Counter(
    word
    for text in train["Text"]
    for word in re.findall(r"[a-z]+(?:'[a-z]+)?", text)
)
print("Training vocabulary:", len(word_counts))
print("Training words found in GloVe:", sum(word in embeddings for word in word_counts))

contributions = Counter()
review_counts = Counter()
training_log_probabilities = model.predict_log_proba(X_train)
training_log_odds = training_log_probabilities[:, 1] - training_log_probabilities[:, 0]

for text, vector, original_score in zip(train["Text"], X_train, training_log_odds):
    counts = Counter(
        word for word in re.findall(r"[a-z]+(?:'[a-z]+)?", text)
        if word in embeddings
    )
    total = sum(counts.values())
    words = [word for word in counts if word_counts[word] >= 5 and counts[word] < total]
    if not words:
        continue
    without_words = np.array([
        (vector * total - embeddings[word] * counts[word]) / (total - counts[word])
        for word in words
    ])
    probabilities = model.predict_log_proba(without_words)
    changes = original_score - (probabilities[:, 1] - probabilities[:, 0])
    for word, change in zip(words, changes):
        contributions[word] += float(change)
        review_counts[word] += 1

word_scores = pd.DataFrame([
    {"Word": word, "Contribution": contributions[word] / review_counts[word],
     "Count": word_counts[word], "Reviews": review_counts[word]}
    for word in contributions if review_counts[word] >= 5
])
positive = word_scores[word_scores["Contribution"] > 0].nlargest(10, "Contribution")
negative = word_scores[word_scores["Contribution"] < 0].nsmallest(10, "Contribution")
positive.to_csv(output / "positive_words.csv", index=False)
negative.to_csv(output / "negative_words.csv", index=False)
print("\nPositive words:\n", positive.to_string(index=False))
print("\nNegative words:\n", negative.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, words, title, color in [
    (axes[0], negative, "Negative word contributions", "tomato"),
    (axes[1], positive, "Positive word contributions", "seagreen"),
]:
    words = words.sort_values("Contribution")
    ax.barh(words["Word"], words["Contribution"], color=color)
    ax.set_title(title)
    ax.set_xlabel("Mean log-odds change: original minus word removed")
fig.tight_layout()
fig.savefig(output / "word_contributions.png", dpi=200)
plt.close(fig)

errors = test[["Text", "Score", "label"]].copy()
errors["prediction"] = predictions
errors = errors[errors["label"] != errors["prediction"]]
examples = pd.concat([
    group.sample(n=min(5, len(group)), random_state=43)
    for _, group in errors.groupby("label")
])
examples.to_csv(output / "misclassified_reviews.csv", index=False)
print("Results saved to:", output)
