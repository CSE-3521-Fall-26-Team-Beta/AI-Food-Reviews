from os import environ
from pathlib import Path
from sys import platform

import pandas as pd
import streamlit as st

from lib.constants import DISTILBERT_LOSS_CURVES_PATH, WORD_EMBEDDINGS_METRICS_PATH
from scripts.p4_word_embeddings import categorize as word_embeddings_gaussian_naive_bayes_categorize, ensure_word_embeddings_report_exists
from scripts.p5_logistic_regression import categorize as tf_idf_logistic_regression_categorize
from scripts.p6_distilbert import categorize as distilbert_categorize

root = Path(__file__).resolve().parent
results = root / 'results'

st.set_page_config(page_title = 'AI Food Reviews', layout = 'wide')
st.title('AI Food Reviews')

prediction, comparison, analysis = st.tabs(
  ['Predict Sentiment', 'Model Comparison', 'DistilBERT Analysis']
)

def print_to_streamlit(text: str):
  if text.count('\n') > 0:
    header, *remainder = text.split('\n')
    st.info(header + '\n' + '\n'.join(map(lambda probability_value: f'- {probability_value}', remainder)))
  else:
    st.info(text)

with prediction:
  with st.form('sentiment_prediction'):
    review = st.text_area(
      'Food review',
      placeholder = 'Enter a food review to compare sentiment predictions.',
    )
    submitted = st.form_submit_button('Predict Sentiment', type = 'primary')
  
  if submitted and not review.strip():
    st.warning('Enter a food review before requesting a prediction.')
  
  models_list = [
    ('Word Embeddings + Gaussian Naive Bayes', word_embeddings_gaussian_naive_bayes_categorize),
    ('TF-IDF + Logistic Regression', tf_idf_logistic_regression_categorize),
  ]
  
  # Checks for if not running in streamlit production deployment
  if not ('HOSTNAME' in environ and environ['HOSTNAME'] == 'streamlit'):
    # DistilBERT crashes Streamlit, so only add it if not in production
    models_list.append(('DistilBERT', distilbert_categorize))
  
  for column, model_info in zip(
    st.columns(len(models_list)), models_list
  ):
    model_name, categorize = model_info
    
    with column:
      st.subheader(model_name)
      
      if submitted and review.strip():
        categorize(
          review,
          print_func = print_to_streamlit,
        )

with comparison:
  ensure_word_embeddings_report_exists()
  st.dataframe(pd.read_csv(WORD_EMBEDDINGS_METRICS_PATH), hide_index = True)

with analysis:
  loss_image = Path(DISTILBERT_LOSS_CURVES_PATH)
  if loss_image.is_file():
    st.image(str(loss_image), caption = 'DistilBERT training and validation loss')
  else:
    st.warning('Cannot find results/distilbert_loss_curves.png')
