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
  classification_report,
  confusion_matrix,
)

root = Path(__file__).resolve().parents[1]
output = root / 'results' / 'word_embeddings'
output.mkdir(parents = True, exist_ok = True)

df = pd.read_csv(root / 'data-bin' / 'group_sample_seed43.csv')
df = df[df['Score'].isin([1, 2, 3, 4, 5])]
df = df.dropna(subset = ['Text']).copy()

labels = ['Negative', 'Neutral', 'Positive']
df['label'] = df['Score'].map({
  1: 'Negative',
  2: 'Negative',
  3: 'Neutral',
  4: 'Positive',
  5: 'Positive',
})

df['Text'] = df['Text'].str.lower()
df['Text'] = df['Text'].str.replace(r'<[^>]+>', ' ', regex = True)
df['Text'] = df['Text'].str.replace(r'\s+', ' ', regex = True).str.strip()
df = df[df['Text'] != '']

label_counts = df.groupby('Text')['label'].nunique()
df = df[df['Text'].isin(label_counts[label_counts == 1].index)]
df = df.drop_duplicates('Text')

embeddings = {}

with open(root / 'data-bin' / 'glove.6B.100d.txt', encoding = 'utf-8') as file:
  for line in file:
    parts = line.split()
    embeddings[parts[0]] = np.array(parts[1:], dtype = np.float32)


def review_vector(text):
  words = re.findall(r'[a-z]+(?:\'[a-z]+)?', text)
  vectors = [embeddings[word] for word in words if word in embeddings]
  
  if len(vectors) == 0:
    return np.zeros(100)
  
  return np.mean(vectors, axis=0)

train, remaining = train_test_split(
  df,
  test_size = 0.30,
  random_state = TRAINING_SEED,
  stratify = df['label'],
)
validation, test = train_test_split(
  remaining,
  test_size = 2 / 3,
  random_state = TRAINING_SEED,
  stratify = remaining['label'],
)

X_train = np.array([review_vector(text) for text in train['Text']])
X_validation = np.array([review_vector(text) for text in validation['Text']])
X_test = np.array([review_vector(text) for text in test['Text']])

y_train = train['label']
y_test = test['label']

best_f1 = -1
for smoothing in [1e-9, 1e-7, 1e-5]:
  candidate = GaussianNB(var_smoothing = smoothing)
  candidate.fit(X_train, y_train)
  score = f1_score(
    validation['label'],
    candidate.predict(X_validation),
    average = 'macro',
    zero_division = 0,
  )
  if score > best_f1:
    best_f1 = score
    model = candidate
predictions = model.predict(X_test)

majority_label = y_train.mode()[0]
baseline = np.full(len(y_test), majority_label)

results = []

def show_results(name, predicted):
  scores = {
    'Model': name,
    'Accuracy': accuracy_score(y_test, predicted),
    'Precision_macro': precision_score(y_test, predicted, labels = labels, average = 'macro', zero_division = 0),
    'Recall_macro': recall_score(y_test, predicted, labels = labels, average = 'macro', zero_division = 0),
    'F1_macro': f1_score(y_test, predicted, labels = labels, average = 'macro', zero_division = 0),
  }
  results.append(scores)
  print('\n' + name)
  for metric in ['Accuracy', 'Precision_macro', 'Recall_macro', 'F1_macro']:
    print(metric + ':', round(scores[metric], 4))


print('Training reviews:', len(train))
print('Validation reviews:', len(validation))
print('Test reviews:', len(test))
print('Features per review:', X_train.shape[1])
print('Selected model:', model)

show_results('Majority baseline', baseline)
show_results('Word embeddings + Gaussian Naive Bayes', predictions)
pd.DataFrame(results).to_csv(output / 'metrics.csv', index = False)
pd.DataFrame(classification_report(
  y_test, predictions, labels = labels, output_dict = True, zero_division = 0,
)).T.to_csv(output / 'classification_report.csv')
pd.DataFrame(
  confusion_matrix(y_test, predictions, labels = labels), index = labels, columns = labels,
).to_csv(output / 'confusion_matrix.csv', index_label = 'Actual / Predicted')

ConfusionMatrixDisplay.from_predictions(
  y_test,
  predictions,
  labels = labels,
  display_labels = labels,
  cmap = 'Blues',
)
plt.tight_layout()
plt.savefig(output / 'confusion_matrix.png', dpi = 200)
plt.close()

word_counts = Counter(
  word
  for text in train['Text']
  for word in re.findall(r'[a-z]+(?:\'[a-z]+)?', text)
)
print('Training vocabulary:', len(word_counts))
print('Training words found in GloVe:', sum(word in embeddings for word in word_counts))

contributions = {}
review_counts = Counter()

def class_scores(vectors):
  probabilities = model.predict_log_proba(vectors)
  return probabilities - (probabilities.sum(axis = 1, keepdims = True) - probabilities) / 2

training_scores = class_scores(X_train)

for text, vector, original_score in zip(train['Text'], X_train, training_scores):
  counts = Counter(
    word for word in re.findall(r'[a-z]+(?:\'[a-z]+)?', text)
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
  for i, label in enumerate(model.classes_)
])
fig, axes = plt.subplots(1, 3, figsize = (16, 6))
for ax, label, color in zip(axes, labels, ['tomato', 'steelblue', 'seagreen']):
  words = word_scores[(word_scores['Sentiment'] == label) & (word_scores['Contribution'] > 0)]
  words = words.nlargest(10, 'Contribution').sort_values('Contribution')
  words.to_csv(output / f'{label.lower()}_words.csv', index = False)
  print(f'\n{label} words:\n', words.to_string(index = False))
  ax.barh(words['Word'], words['Contribution'], color = color)
  ax.set_title(f'Words associated with {label.lower()}')
  ax.set_xlabel('Mean change in relative class log score\n(original minus word removed)')
fig.suptitle('GloVe + Gaussian NB: training-review word contributions')
fig.tight_layout()
fig.savefig(output / 'word_contributions.png', dpi = 200)
plt.close(fig)

errors = test[['Text', 'Score', 'label']].copy()
errors['prediction'] = predictions
errors = errors[errors['label'] != errors['prediction']]
examples = pd.concat([
  group.sample(
    n = min(3, len(group)),
    random_state = TRAINING_SEED,
  )
  for _, group in errors.groupby('label')
])
examples.to_csv(output / 'misclassified_reviews.csv', index = False)
print('Results saved to:', output)
