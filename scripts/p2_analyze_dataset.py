from os.path import exists

import pandas as pd
import matplotlib.pyplot as plt

from lib.constants import DATASET_ANALYSIS_PATH, SAMPLE_CSV_PATH, SCORE_DISTRIBUTION_PATH
from lib.function_helpers import lazy_evaluation

get_df = lazy_evaluation(lambda: pd.read_csv(SAMPLE_CSV_PATH))

if not exists(DATASET_ANALYSIS_PATH):
  print('Performing dataset analysis...')
  
  with open(DATASET_ANALYSIS_PATH, 'w') as f:
    df = get_df().copy()
    
    print('\nTotal Reviews:', len(df), file = f)
    
    print('\nScore Distribution:', file = f)
    print(df['Score'].value_counts().sort_index(), file = f)
    
    df['Sentiment'] = df['Score'].map({
      1: 'Negative',
      2: 'Negative',
      3: 'Neutral',
      4: 'Positive',
      5: 'Positive',
    })
    
    print('\nSentiment Distribution:')
    counts = df['Sentiment'].value_counts()
    
    for label in ['Negative', 'Neutral', 'Positive']:
      count = counts.get(label, 0)
      print(f'{label}: {count} ({count / len(df) * 100:.2f}%)')
    
    print('\nAverage Review Length:')
    print(round(df['Text'].fillna('').str.split().str.len().mean(), 2), 'words')
    
    print('\nMissing Values:')
    print(df.isnull().sum())
    
    print('\nDuplicate Review Texts:')
    print(df['Text'].duplicated().sum())
    
    print('\nDuplicate Review Records:')
    print(df.duplicated(
      subset = ['ProductId', 'UserId', 'Score', 'Summary', 'Text'],
    ).sum())
else:
  print('Dataset analysis already performed')

scores = df['Score'].value_counts().sort_index()

plt.bar(scores.index, scores.values)
plt.title('Group Beta - Review Score Distribution')
plt.xlabel('Star Rating')
plt.ylabel('Number of Reviews')
plt.xticks([1, 2, 3, 4, 5])
plt.tight_layout()
plt.savefig(SCORE_DISTRIBUTION_PATH, dpi = 300)

print(f'\nChart saved to {SCORE_DISTRIBUTION_PATH}')
