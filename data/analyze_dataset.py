
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data-bin/group_sample_seed43.csv")

print("\nTotal Reviews:", len(df))

print("\nScore Distribution:")
print(df["Score"].value_counts().sort_index())

df["Sentiment"] = df["Score"].map({
    1: "Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Positive"
})

print("\nSentiment Distribution:")
counts = df["Sentiment"].value_counts()

for label in ["Negative", "Neutral", "Positive"]:
    count = counts.get(label, 0)
    print(f"{label}: {count} ({count / len(df) * 100:.2f}%)")

print("\nAverage Review Length:")
print(round(df["Text"].fillna("").str.split().str.len().mean(), 2), "words")

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Review Texts:")
print(df["Text"].duplicated().sum())

print("\nDuplicate Review Records:")
print(df.duplicated(
    subset=["ProductId", "UserId", "Score", "Summary", "Text"]
).sum())

scores = df["Score"].value_counts().sort_index()

plt.bar(scores.index, scores.values)
plt.title("Group Beta - Review Score Distribution")
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.xticks([1, 2, 3, 4, 5])
plt.tight_layout()
plt.savefig("data-bin/score_distribution.png", dpi=300)

print("\nChart saved to data-bin/score_distribution.png")
