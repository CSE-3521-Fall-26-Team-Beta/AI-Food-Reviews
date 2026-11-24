'Generates logistic regression classical model. Run from the repository root with "python -m scripts.p5_logistic_regression" to train and test, and [TODO] "python -m scripts.p5_logistic_regression --review <your review text in quotes>" to train and predict.'

import argparse
from os.path import exists

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from lib.constants import LOGISTIC_REGRESSION_REPORT_PATH
from lib.ml_helpers import print_and_write
from scripts.p3_feature_engineering import perform_feature_engineering

_X_test = None
_X_train = None
_test_df = None
_train_df = None
_transform_data = None
_model = None

def _init_model_vars():
  global _X_test, _X_train, _test_df, _train_df, _transform_data, _model
  
  if _X_test == None:
    _X_test, _X_train, _test_df, _train_df, _transform_data = perform_feature_engineering()
    
    # Give each sentiment equal weight even though most reviews are positive.
    _model = LogisticRegression(class_weight = 'balanced', max_iter = 300)
    _model.fit(_X_train, _train_df['Sentiment'])

def test():
  if not exists(LOGISTIC_REGRESSION_REPORT_PATH):
    print('Calculating logistic regression...')
    
    with open(LOGISTIC_REGRESSION_REPORT_PATH, 'w', newline = '') as f:
      _init_model_vars()
      
      predictions = _model.predict(_X_test)
      print_and_write('Logistic regression on held-out reviews', file = f)
      print_and_write(classification_report(_test_df['Sentiment'], predictions, digits = 4), file = f)
    
    print(f'Logistic regression report saved to {LOGISTIC_REGRESSION_REPORT_PATH}')
  else:
    print('Logistic regression report already calculated')

def categorize(review_text, print_func = print):
  _init_model_vars()
  
  prediction = _model.predict(_transform_data([review_text]))[0]
  raw_prediction = _model.decision_function(_transform_data([review_text]))[0]
  
  if print_func != None:
    print_func(f'Prediction: {prediction}')
    print_func(f'Raw Prediction: {raw_prediction}')
  else:
    return prediction

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = __doc__)
  parser.add_argument('--review', help = 'Classify a review with the saved model')
  args = parser.parse_args()
  
  if args.review is not None:
    categorize(args.review)
  else:
    test()
