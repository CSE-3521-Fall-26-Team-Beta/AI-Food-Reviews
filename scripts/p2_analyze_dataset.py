from os.path import exists

import pandas as pd
import matplotlib.pyplot as plt

from lib.constants import DATASET_ANALYSIS_PATH, SCORE_DISTRIBUTION_PATH
from lib.generate_group_sample import get_sample_df

if not exists(DATASET_ANALYSIS_PATH):
  print('Performing dataset analysis...')
  
  with open(DATASET_ANALYSIS_PATH, 'w', newline = '') as f:
    df = get_sample_df()
    
    print('Total Reviews:', len(df), file = f)
    
    print('\nScore Distribution:', file = f)
    print(df['Score'].value_counts().sort_index(), file = f)
    
    df['Sentiment'] = df['Score'].map({
      1: 'Negative',
      2: 'Negative',
      3: 'Neutral',
      4: 'Positive',
      5: 'Positive',
    })
    
    print('\nSentiment Distribution:', file = f)
    counts = df['Sentiment'].value_counts()
    
    for label in ['Negative', 'Neutral', 'Positive']:
      count = counts.get(label, 0)
      print(f'{label}: {count} ({count / len(df) * 100:.2f}%)', file = f)
    
    print('\nAverage Review Length:', file = f)
    print(round(df['Text'].fillna('').str.split().str.len().mean(), 2), 'words', file = f)
    
    print('\nMissing Values:', file = f)
    print(df.isnull().sum(), file = f)
    
    print('\nDuplicate Review Texts:', file = f)
    print(df['Text'].duplicated().sum(), file = f)
    
    print('\nDuplicate Review Records:', file = f)
    print(
      df.duplicated(
        subset = ['ProductId', 'UserId', 'Score', 'Summary', 'Text'],
      ).sum(),
      file = f,
    )
  
  print(f'Dataset analysis saved to {DATASET_ANALYSIS_PATH}')
else:
  print('Dataset analysis already performed')

if not exists(SCORE_DISTRIBUTION_PATH):
  print('Generating score distribution chart...')
  
  df = get_sample_df()
  
  scores = df['Score'].value_counts().sort_index()
  
  plt.bar(scores.index, scores.values)
  plt.title('Group Beta - Review Score Distribution')
  plt.xlabel('Star Rating')
  plt.ylabel('Number of Reviews')
  plt.xticks([1, 2, 3, 4, 5])
  plt.tight_layout()
  plt.savefig(SCORE_DISTRIBUTION_PATH, dpi = 300)
  
  print(f'Score distribution chart saved to {SCORE_DISTRIBUTION_PATH}')
else:
  print('Score distribution chart already created')
