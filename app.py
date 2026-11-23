from pathlib import Path

import pandas as pd
import streamlit as st

root = Path(__file__).resolve().parent
results = root / "results"

st.set_page_config(page_title="AI Food Reviews", layout="wide")
st.title("AI Food Reviews")

prediction, comparison, analysis = st.tabs(
  ["Predict Sentiment", "Model Comparison", "DistilBERT Analysis"]
)

with prediction:
  with st.form("sentiment_prediction"):
    review = st.text_area(
      "Food review",
      placeholder="Enter a food review to compare sentiment predictions.",
    )
    submitted = st.form_submit_button("Predict Sentiment", type="primary")
  
  if submitted and not review.strip():
    st.warning("Enter a food review before requesting a prediction.")
  
  for column, model_name in zip(
    st.columns(2), ("Gaussian Naive Bayes", "DistilBERT")
  ):
    with column:
      st.subheader(model_name)
      st.info("Predictions will appear here once this model is connected.")

with comparison:
  metrics_file = results / "word_embeddings" / "metrics.csv"
  if metrics_file.is_file():
    st.dataframe(pd.read_csv(metrics_file), hide_index=True)
  else:
    st.warning("Cannot find results/word_embeddings/metrics.csv")

with analysis:
  loss_image = results / "distilbert_loss_curves.png"
  if loss_image.is_file():
    st.image(str(loss_image), caption="DistilBERT training and validation loss")
  else:
    st.warning("Cannot find results/distilbert_loss_curves.png")
