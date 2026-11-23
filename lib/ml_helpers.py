from pandas import DataFrame

from .generate_group_sample import get_sample_df

def get_annotated_sample_df() -> DataFrame:
  df = get_sample_df()
  
  # Assign sentiment labels
  df['Sentiment'] = df['Score'].map({
    1: 'Negative',
    2: 'Negative',
    3: 'Neutral',
    4: 'Positive',
    5: 'Positive',
  })
  
  return df
