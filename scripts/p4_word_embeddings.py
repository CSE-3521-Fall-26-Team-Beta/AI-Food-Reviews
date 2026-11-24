'Generates word embeddings and Gaussian Naive Bayes classical model. Run from the repository root with "python -m scripts.p4_word_embeddings" to train, and "python -m scripts.p4_word_embeddings --review <your review text in quotes>" to predict.'

import argparse
import re
from collections import Counter
from pathlib import Path
from os import makedirs
from os.path import exists

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
  accuracy_score,
  precision_score,
  recall_score,
  f1_score,
  ConfusionMatrixDisplay,
  classification_report,
  confusion_matrix,
)

from lib.constants import (
  GLOVE_PATH,
  RESULTS_DIR,
  TRAINING_SEED,
  WORD_EMBEDDINGS_CLASSIFICATION_REPORT_PATH,
  WORD_EMBEDDINGS_CONFUSION_MATRIX_DATA_PATH,
  WORD_EMBEDDINGS_CONFUSION_MATRIX_IMAGE_PATH,
  WORD_EMBEDDINGS_DIR,
  WORD_EMBEDDINGS_LABELED_WORDS_DIR,
  WORD_EMBEDDINGS_METRICS_PATH,
  WORD_EMBEDDINGS_MISCLASSIFIED_REVIEWS_PATH,
  WORD_EMBEDDINGS_REPORT_PATH,
  WORD_EMBEDDINGS_WORD_CONTRIBUTIONS_IMAGE_PATH,
)
from lib.glove_download import ensure_glove_exists
from lib.ml_helpers import get_annotated_sample_df, print_and_write

_LABELS = ['Negative', 'Neutral', 'Positive']

_embeddings = None
_train_dataset = None
_validation_dataset = None
_test_dataset = None
_X_train = None
_X_test = None
_y_train = None
_y_test = None
_model = None

def _transform_data(text):
  words = re.findall(r'[a-z]+(?:\'[a-z]+)?', text)
  vectors = [_embeddings[word] for word in words if word in _embeddings]
  
  if len(vectors) == 0:
    return np.zeros(100)
  
  return np.mean(vectors, axis = 0)

def _train():
  global _embeddings, _train_dataset, _validation_dataset, _test_dataset, _X_train, _X_test, _y_train, _y_test, _model
  
  # Note: this previously used the entire dataset not just the sample
  # Tweaked to only use the sample
  df = get_annotated_sample_df()
  df = df[df['Score'].isin([1, 2, 3, 4, 5])]
  df = df.dropna(subset = ['Text']).copy()
  
  df['Text'] = df['Text'].str.lower()
  df['Text'] = df['Text'].str.replace(r'<[^>]+>', ' ', regex = True)
  df['Text'] = df['Text'].str.replace(r'\s+', ' ', regex = True).str.strip()
  df = df[df['Text'] != '']
  
  label_counts = df.groupby('Text')['Sentiment'].nunique()
  df = df[df['Text'].isin(label_counts[label_counts == 1].index)]
  df = df.drop_duplicates('Text')
  
  _embeddings = {}
  
  ensure_glove_exists()
  with open(GLOVE_PATH, encoding = 'utf-8') as file:
    for line in file:
      parts = line.split()
      _embeddings[parts[0]] = np.array(parts[1:], dtype = np.float32)
  
  _train_dataset, remaining = train_test_split(
    df,
    test_size = 0.30,
    random_state = TRAINING_SEED,
    stratify = df['Sentiment'],
  )
  _validation_dataset, _test_dataset = train_test_split(
    remaining,
    test_size = 2 / 3,
    random_state = TRAINING_SEED,
    stratify = remaining['Sentiment'],
  )
  
  _X_train = np.array([_transform_data(text) for text in _train_dataset['Text']])
  X_validation = np.array([_transform_data(text) for text in _validation_dataset['Text']])
  _X_test = np.array([_transform_data(text) for text in _test_dataset['Text']])
  
  _y_train = _train_dataset['Sentiment']
  _y_test = _test_dataset['Sentiment']
  
  best_f1 = -1
  counts = _y_train.value_counts().sort_index().to_numpy()
  for prior_power in [1, 0.5, 0]:
    priors = counts ** prior_power
    candidate = make_pipeline(
      LinearDiscriminantAnalysis(),
      GaussianNB(priors = priors / priors.sum()),
    )
    candidate.fit(_X_train, _y_train)
    score = f1_score(
      _validation_dataset['Sentiment'],
      candidate.predict(X_validation),
      average = 'macro',
      zero_division = 0,
    )
    if score > best_f1:
      best_f1 = score
      _model = candidate

def _init_model_vars():
  if _embeddings == None:
    _train()

def ensure_word_embeddings_report_exists(force = False):
  if force or not exists(WORD_EMBEDDINGS_REPORT_PATH):
    print('Calculating word embeddings...')
    
    with open(WORD_EMBEDDINGS_REPORT_PATH, 'w', newline = '') as f:
      output = WORD_EMBEDDINGS_DIR
      Path(output).mkdir(parents = True, exist_ok = True)
      
      _init_model_vars()
      
      predictions = _model.predict(_X_test)
      
      majority_label = _y_train.mode()[0]
      baseline = np.full(len(_y_test), majority_label)
      
      results = []
      
      def show_results(name, predicted):
        scores = {
          'Model': name,
          'Accuracy': accuracy_score(_y_test, predicted),
          'Precision_macro': precision_score(_y_test, predicted, labels = _LABELS, average = 'macro', zero_division = 0),
          'Recall_macro': recall_score(_y_test, predicted, labels = _LABELS, average = 'macro', zero_division = 0),
          'F1_macro': f1_score(_y_test, predicted, labels = _LABELS, average = 'macro', zero_division = 0),
        }
        results.append(scores)
        print_and_write('\n' + name, file = f)
        for metric in ['Accuracy', 'Precision_macro', 'Recall_macro', 'F1_macro']:
          print_and_write(metric + ':', round(scores[metric], 4), file = f)
      
      print_and_write('Training reviews:', len(_train_dataset), file = f)
      print_and_write('Validation reviews:', len(_validation_dataset), file = f)
      print_and_write('Test reviews:', len(_test_dataset), file = f)
      print_and_write('Features per review:', _X_train.shape[1], file = f)
      print_and_write('Selected model:', _model, file = f)
      
      show_results('Majority baseline', baseline)
      show_results('Word embeddings + Gaussian Naive Bayes', predictions)
      makedirs(RESULTS_DIR, exist_ok = True)
      pd.DataFrame(results).to_csv(WORD_EMBEDDINGS_METRICS_PATH, index = False, lineterminator = '\n')
      pd.DataFrame(classification_report(
        _y_test, predictions, labels = _LABELS, output_dict = True, zero_division = 0,
      )).T.to_csv(WORD_EMBEDDINGS_CLASSIFICATION_REPORT_PATH, lineterminator = '\n')
      pd.DataFrame(
        confusion_matrix(_y_test, predictions, labels = _LABELS), index = _LABELS, columns = _LABELS,
      ).to_csv(WORD_EMBEDDINGS_CONFUSION_MATRIX_DATA_PATH, index_label = 'Actual / Predicted', lineterminator = '\n')
      
      ConfusionMatrixDisplay.from_predictions(
        _y_test,
        predictions,
        labels = _LABELS,
        display_labels = _LABELS,
        cmap = 'Blues',
      )
      plt.tight_layout()
      plt.savefig(WORD_EMBEDDINGS_CONFUSION_MATRIX_IMAGE_PATH, dpi = 200)
      plt.close()
      
      word_counts = Counter(
        word
        for text in _train_dataset['Text']
        for word in re.findall(r'[a-z]+(?:\'[a-z]+)?', text)
      )
      print_and_write('Training vocabulary:', len(word_counts), file = f)
      print_and_write('Training words found in GloVe:', sum(word in _embeddings for word in word_counts), file = f)
      
      contributions = {}
      review_counts = Counter()
      
      def class_scores(vectors):
        probabilities = _model.predict_log_proba(vectors)
        return probabilities - (probabilities.sum(axis = 1, keepdims = True) - probabilities) / 2
      
      training_scores = class_scores(_X_train)
      
      for text, vector, original_score in zip(_train_dataset['Text'], _X_train, training_scores):
        counts = Counter(
          word for word in re.findall(r'[a-z]+(?:\'[a-z]+)?', text)
          if word in _embeddings
        )
        total = sum(counts.values())
        words = [word for word in counts if word_counts[word] >= 5 and counts[word] < total]
        if not words:
          continue
        without_words = np.array([
          (vector * total - _embeddings[word] * counts[word]) / (total - counts[word])
          for word in words
        ])
        changes = original_score - class_scores(without_words)
        for word, change in zip(words, changes):
          if word not in contributions:
            contributions[word] = np.zeros(3)
          contributions[word] += change
          review_counts[word] += 1
      
      word_scores = pd.DataFrame([
        {
          'Word': word,
          'Sentiment': label,
          'Contribution': contributions[word][i] / review_counts[word],
          'Count': word_counts[word],
          'Reviews': review_counts[word],
        }
        for word in contributions if review_counts[word] >= 5
        for i, label in enumerate(_model.classes_)
      ])
      makedirs(WORD_EMBEDDINGS_LABELED_WORDS_DIR, exist_ok = True)
      fig, axes = plt.subplots(1, 3, figsize = (16, 6))
      for ax, label, color in zip(axes, _LABELS, ['tomato', 'steelblue', 'seagreen']):
        words = word_scores[(word_scores['Sentiment'] == label) & (word_scores['Contribution'] > 0)]
        words = words.nlargest(10, 'Contribution').sort_values('Contribution')
        words.to_csv(f'{WORD_EMBEDDINGS_LABELED_WORDS_DIR}/{label.lower()}_words.csv', index = False, lineterminator = '\n')
        print_and_write(f'\n{label} words:\n', words.to_string(index = False), file = f)
        ax.barh(words['Word'], words['Contribution'], color = color)
        ax.set_title(f'Words associated with {label.lower()}')
        ax.set_xlabel('Mean change in relative class log score\n(original minus word removed)')
      fig.suptitle('GloVe + Gaussian NB: training-review word contributions')
      fig.tight_layout()
      fig.savefig(WORD_EMBEDDINGS_WORD_CONTRIBUTIONS_IMAGE_PATH, dpi = 200)
      plt.close(fig)
      
      errors = _test_dataset[['Text', 'Score', 'Sentiment']].copy()
      errors['prediction'] = predictions
      errors = errors[errors['Sentiment'] != errors['prediction']]
      examples = pd.concat([
        group.sample(
          n = min(3, len(group)),
          random_state = TRAINING_SEED,
        )
        for _, group in errors.groupby('Sentiment')
      ])
      examples.to_csv(WORD_EMBEDDINGS_MISCLASSIFIED_REVIEWS_PATH, index = False, lineterminator = '\n')
    
    print()
    print('Results saved to:', output)
    print(f'Word embedding report saved to {WORD_EMBEDDINGS_REPORT_PATH} and other files also saved')
    
    return True
  else:
    return False

def test():
  ensure_word_embeddings_report_exists(force = True)

def categorize(review_text, print_func = print):
  _init_model_vars()
  text = re.sub(r'<[^>]+>', ' ', review_text.lower())
  vector = _transform_data(text)
  if not np.any(vector):
    if print_func is not None:
      print_func('Enter a review with words recognized by the GloVe model.')
      return
    return {'prediction': None, 'probabilities': {}}

  prediction = _model.predict([vector])[0].lower()
  prediction_probabilities = zip(
    (label.lower() for label in _model.classes_),
    _model.predict_proba([vector])[0]
  )
  
  if print_func != None:
    print_func(f'Prediction: {prediction.title()}')
    
    print_func(
      'Prediction Probabilities:\n' + 
      '\n'.join(f'{label.title()}: {probability:.1%}' for label, probability in prediction_probabilities)
    )
  else:
    return {
      'prediction': prediction,
      'probabilities': dict(prediction_probabilities),
    }

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = __doc__)
  parser.add_argument('--review', help = 'Classify a review with the saved model')
  args = parser.parse_args()
  
  if args.review is not None:
    categorize(args.review)
  else:
    test()
