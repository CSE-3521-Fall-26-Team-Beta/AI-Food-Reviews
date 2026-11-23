"""Run from the repository root with python -m data.logistic_regression."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from data.feature_engineering import X_test, X_train, test_df, train_df

if __name__ == "__main__":
  # Give each sentiment equal weight even though most reviews are positive.
  model = LogisticRegression(class_weight="balanced", max_iter=300)
  model.fit(X_train, train_df["Sentiment"])
  
  predictions = model.predict(X_test)
  print("\nLogistic regression on held-out reviews")
  print(classification_report(test_df["Sentiment"], predictions, digits=4))
