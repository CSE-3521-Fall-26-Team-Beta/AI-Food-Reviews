'Generates logistic regression classical model. Run from the repository root with "python -m scripts.p5_logistic_regression" to train, and [TODO] "python -m scripts.p5_logistic_regression --review <your review text in quotes>" to predict.'

from os.path import exists

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from lib.constants import LOGISTIC_REGRESSION_REPORT_PATH
from lib.ml_helpers import print_and_write
from scripts.p3_feature_engineering import X_test, X_train, test_df, train_df

if __name__ == '__main__':
  if not exists(LOGISTIC_REGRESSION_REPORT_PATH):
    print('Calculating logistic regression...')
    
    with open(LOGISTIC_REGRESSION_REPORT_PATH, 'w', newline = '') as f:
      # Give each sentiment equal weight even though most reviews are positive.
      model = LogisticRegression(class_weight = 'balanced', max_iter = 300)
      model.fit(X_train, train_df['Sentiment'])
      
      predictions = model.predict(X_test)
      print_and_write('Logistic regression on held-out reviews', file = f)
      print_and_write(classification_report(test_df['Sentiment'], predictions, digits = 4), file = f)
  
  print(f'Logistic regression report saved to {LOGISTIC_REGRESSION_REPORT_PATH}')
else:
  print('Logistic regression report already calculated')
