from os.path import exists
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from lib.constants import FEATURE_ENGINEERING_REPORT_PATH, TRAINING_SEED
from lib.generate_group_sample import get_sample_df

if not exists(FEATURE_ENGINEERING_REPORT_PATH):
  print('Creating feature engineering report...')
  
  with open(FEATURE_ENGINEERING_REPORT_PATH, 'w') as f:
    # Load Group Beta dataset
    df = get_sample_df()
    
    # Clean review text
    def clean_text(text):
      text = str(text)
      text = re.sub(r'<[^>]+>', ' ', text)
      text = re.sub(r'\s+', ' ', text)
      return text.strip()
    
    df['Text'] = df['Text'].fillna('').apply(clean_text)
    
    # Assign sentiment labels
    df['Sentiment'] = df['Score'].map({
      1: 'Negative',
      2: 'Negative',
      3: 'Neutral',
      4: 'Positive',
      5: 'Positive',
    })
    
    # Check for conflicting labels in identical reviews
    conflicts = (
      df.groupby('Text')['Sentiment']
        .nunique()
        .gt(1)
        .sum()
    )
    
    print('\nConflicting Sentiment Labels:', file = f)
    print('Review texts with conflicting labels:', conflicts, file = f)
    
    # Remove duplicate reviews
    original_count = len(df)
    
    conflicting_texts = (
      df.groupby('Text')['Sentiment']
        .nunique()
    )
    
    conflicting_texts = conflicting_texts[conflicting_texts > 1].index
    
    df = df[~df['Text'].isin(conflicting_texts)].copy()
    df = df.drop_duplicates(subset=['Text']).copy()
    
    removed_count = original_count - len(df)
    
    print('\nDuplicate Review Cleaning:', file = f)
    print('Original reviews:', original_count, file = f)
    print('Duplicates removed:', removed_count, file = f)
    print('Reviews remaining:', len(df), file = f)
    print('Duplicates remaining:', df['Text'].duplicated().sum(), file = f)
    
    # Display updated sentiment distribution
    print('\nSentiment Distribution After Cleaning:', file = f)
    print(df['Sentiment'].value_counts(), file = f)
    
    # Split data: 80% training, 10% validation, 10% testing
    train_df, temp_df = train_test_split(
      df,
      test_size = 0.20,
      random_state = TRAINING_SEED,
      stratify = df['Sentiment'],
    )
    
    val_df, test_df = train_test_split(
      temp_df,
      test_size = 0.50,
      random_state = TRAINING_SEED,
      stratify = temp_df['Sentiment'],
    )
    
    # Configure TF-IDF
    vectorizer = TfidfVectorizer(
      lowercase = True,
      stop_words = 'english',
      max_features = 10000,
      min_df = 5,
      max_df = 0.95,
      ngram_range = (1, 2),
      sublinear_tf = True,
    )
    
    # Fit only on training data to prevent data leakage
    X_train = vectorizer.fit_transform(train_df['Text'])
    X_val = vectorizer.transform(val_df['Text'])
    X_test = vectorizer.transform(test_df['Text'])
    
    # Display feature engineering results
    print('\nTF-IDF FEATURE ENGINEERING REPORT', file = f)
    
    print('\nDataset Splits:', file = f)
    print('Training:', len(train_df), file = f)
    print('Validation:', len(val_df), file = f)
    print('Testing:', len(test_df), file = f)
    
    print('\nFeature Matrix Shapes:', file = f)
    print('Training:', X_train.shape, file = f)
    print('Validation:', X_val.shape, file = f)
    print('Testing:', X_test.shape, file = f)
    
    print('\nVocabulary Size:', file = f)
    print(len(vectorizer.get_feature_names_out()), file = f)
    
    print('\nSample TF-IDF Features:', file = f)
    print(vectorizer.get_feature_names_out()[:20], file = f)
    
    # Display top weighted features
    print('\nTop Weighted Features (First Training Review):', file = f)
    
    feature_names = vectorizer.get_feature_names_out()
    row = X_train[0]
    
    top_indices = row.toarray().flatten().argsort()[-10:][::-1]
    
    for i in top_indices:
      if row[0, i] > 0:
        print(f'{feature_names[i]}: {row[0, i]:.4f}', file = f)
  
  print(f'Feature engineering report saved to {FEATURE_ENGINEERING_REPORT_PATH}')
else:
  print('Feature engineering report already created')
