import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from lib.constants import FEATURE_ENGINEERING_REPORT_PATH, TRAINING_SEED
from lib.ml_helpers import get_annotated_sample_df


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def perform_feature_engineering(report_file=None):
    df = get_annotated_sample_df().copy()
    original_count = len(df)

    df = df[df["Score"].isin([1, 2, 3, 4, 5])].copy()
    df["Text"] = df["Text"].apply(clean_text)

    empty_count = int((df["Text"] == "").sum())
    df = df[df["Text"] != ""].copy()

    label_counts = df.groupby("Text")["Sentiment"].nunique()
    conflicting_texts = label_counts[label_counts > 1].index

    conflicting_count = len(conflicting_texts)
    conflicting_rows = int(df["Text"].isin(conflicting_texts).sum())

    df = df[~df["Text"].isin(conflicting_texts)].copy()

    duplicate_count = int(df["Text"].duplicated().sum())
    df = df.drop_duplicates(subset=["Text"], keep="first").copy()

    final_count = len(df)
    removed_count = original_count - final_count

    if report_file is not None:
        print("DATA CLEANING REPORT", file=report_file)
        print("\nOriginal reviews:", original_count, file=report_file)
        print("Empty reviews removed:", empty_count, file=report_file)
        print("Conflicting review texts:", conflicting_count, file=report_file)
        print("Rows with conflicting labels:", conflicting_rows, file=report_file)
        print("Duplicate reviews removed:", duplicate_count, file=report_file)
        print("Total rows removed:", removed_count, file=report_file)
        print("Reviews remaining:", final_count, file=report_file)
        print(
            "Duplicates remaining:",
            df["Text"].duplicated().sum(),
            file=report_file
        )

        print("\nSentiment Distribution After Cleaning:", file=report_file)
        print(df["Sentiment"].value_counts(), file=report_file)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=TRAINING_SEED,
        stratify=df["Sentiment"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=2 / 3,
        random_state=TRAINING_SEED,
        stratify=temp_df["Sentiment"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=10000,
        min_df=5,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X_train = csr_matrix(vectorizer.fit_transform(train_df["Text"]))
    X_val = csr_matrix(vectorizer.transform(val_df["Text"]))
    X_test = csr_matrix(vectorizer.transform(test_df["Text"]))

    def transform_data(review_text):
        if isinstance(review_text, str):
            review_text = [review_text]

        cleaned_reviews = [clean_text(text) for text in review_text]

        return vectorizer.transform(cleaned_reviews)

    if report_file is not None:
        print("\nTF-IDF FEATURE ENGINEERING REPORT", file=report_file)

        print("\nDataset Splits:", file=report_file)
        print("Training:", len(train_df), file=report_file)
        print("Validation:", len(val_df), file=report_file)
        print("Testing:", len(test_df), file=report_file)

        print("\nFeature Matrix Shapes:", file=report_file)
        print("Training:", X_train.shape, file=report_file)
        print("Validation:", X_val.shape, file=report_file)
        print("Testing:", X_test.shape, file=report_file)

        feature_names = vectorizer.get_feature_names_out()

        print("\nVocabulary Size:", len(feature_names), file=report_file)

        print("\nTF-IDF Configuration:", file=report_file)
        print("Maximum features: 10000", file=report_file)
        print("Minimum document frequency: 5", file=report_file)
        print("Maximum document frequency: 0.95", file=report_file)
        print("N-gram range: (1, 2)", file=report_file)
        print("Stop words: English", file=report_file)
        print("Sublinear term frequency: Enabled", file=report_file)

        print("\nSample TF-IDF Features:", file=report_file)
        print(feature_names[:20], file=report_file)

        average_weights = (
            np.asarray(X_train.sum(axis=0)).ravel()
            / X_train.shape[0]
        )

        top_indices = np.argsort(average_weights)[-10:][::-1]

        print("\nTop 10 Features by Average TF-IDF Weight:", file=report_file)

        for i in top_indices:
            print(
                f"{feature_names[i]}: {average_weights[i]:.4f}",
                file=report_file
            )

        total_values = X_train.shape[0] * X_train.shape[1]
        sparsity = 1 - (X_train.nnz / total_values)

        print(
            "\nTraining Matrix Sparsity:",
            f"{sparsity * 100:.2f}%",
            file=report_file
        )

    return X_test, X_train, test_df, train_df, transform_data


if __name__ == "__main__":
    report_path = Path(FEATURE_ENGINEERING_REPORT_PATH)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("Creating feature engineering report...")

    with open(report_path, "w", encoding="utf-8") as report_file:
        perform_feature_engineering(report_file=report_file)

    print(f"Feature engineering report saved to {report_path}")
