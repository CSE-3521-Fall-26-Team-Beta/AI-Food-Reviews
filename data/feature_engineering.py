import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Group Beta dataset
df = pd.read_csv("data-bin/group_sample_seed43.csv")

# Clean review text
def clean_text(text):
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["Text"] = df["Text"].fillna("").apply(clean_text)

# Assign sentiment labels
df["Sentiment"] = df["Score"].map({
    1: "Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Positive"
})

# Check for conflicting labels in identical reviews
conflicts = (
    df.groupby("Text")["Sentiment"]
    .nunique()
    .gt(1)
    .sum()
)

print("\nConflicting Sentiment Labels:")
print("Review texts with conflicting labels:", conflicts)

# Remove duplicate reviews
original_count = len(df)

conflicting_texts = (
    df.groupby("Text")["Sentiment"]
    .nunique()
)

conflicting_texts = conflicting_texts[conflicting_texts > 1].index

df = df[~df["Text"].isin(conflicting_texts)].copy()
df = df.drop_duplicates(subset=["Text"]).copy()

removed_count = original_count - len(df)

print("\nDuplicate Review Cleaning:")
print("Original reviews:", original_count)
print("Duplicates removed:", removed_count)
print("Reviews remaining:", len(df))
print("Duplicates remaining:", df["Text"].duplicated().sum())

# Display updated sentiment distribution
print("\nSentiment Distribution After Cleaning:")
print(df["Sentiment"].value_counts())

# Split data: 80% training, 10% validation, 10% testing
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=43,
    stratify=df["Sentiment"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=43,
    stratify=temp_df["Sentiment"]
)

# Configure TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=10000,
    min_df=5,
    max_df=0.95,
    ngram_range=(1, 2),
    sublinear_tf=True
)

# Fit only on training data to prevent data leakage
X_train = vectorizer.fit_transform(train_df["Text"])
X_val = vectorizer.transform(val_df["Text"])
X_test = vectorizer.transform(test_df["Text"])

# Display feature engineering results
print("\nTF-IDF FEATURE ENGINEERING REPORT")

print("\nDataset Splits:")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

print("\nFeature Matrix Shapes:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nVocabulary Size:")
print(len(vectorizer.get_feature_names_out()))

print("\nSample TF-IDF Features:")
print(vectorizer.get_feature_names_out()[:20])

# Display top weighted features
print("\nTop Weighted Features (First Training Review):")

feature_names = vectorizer.get_feature_names_out()
row = X_train[0]

top_indices = row.toarray().flatten().argsort()[-10:][::-1]

for i in top_indices:
    if row[0, i] > 0:
        print(f"{feature_names[i]}: {row[0, i]:.4f}")
