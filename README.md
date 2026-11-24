# AI-Food-Reviews

This is the GitHub repository for the AI food review project for CSE 3521.

# Team Members

- [Ethan Boes](https://github.com/CaptainDeuce)
- [Max Wuchter](https://github.com/MWuchter)
- [Haodong Qiu](https://github.com/meteorfallground-ops)
- [Santosh Schaefer](https://github.com/santosh-sch)
- [Sulman Abdullahi](https://github.com/Sulmanabdullahi)

# Project Description

Our project will use the Amazon Fine Food Reviews dataset to build and evaluate a sentiment analysis system. We plan to use a shared sample of 50,000 reviews and convert the original star ratings into three sentiment categories: negative (1–2 stars), neutral (3 stars), and positive (4–5 stars).

We will preprocess the review text and apply TF-IDF and word embeddings as our features engineering methods to convert textual information into numerical representations. Next, we will implement two classical machine learning models, Naive Bayes and Gaussian Mixture Model (GMM), and compare their performance with a pre-trained transformer model, DistilBERT. We will evaluate the models using accuracy, precision, recall and F1-score.

Finally, we will develop and deploy a Streamlit web application that allows users to enter their own review text and view sentiment predictions from at least two trained models. Through this project, we aim to understand how different feature representations and model architectures influence the accuracy and efficiency of sentiment classification.

# Run GloVe + Gaussian Naive Bayes

With the sample CSV and `glove.6B.100d.txt` in `data-bin/`, run from the repo root:

```powershell
python -m scripts.p4_word_embeddings
```

The script classifies reviews as Negative (1–2 stars), Neutral (3), or Positive
(4–5), using a 70/10/20 train/validation/test split with seed 43. Variance smoothing
is selected using validation macro F1. Metrics, the three-class confusion matrix,
word-contribution chart, and misclassified examples are saved in
`results/word_embeddings/`. Precision, recall, and F1 use macro averaging.

# Full results regeneration

To recalculate everything, delete the `models` and `results` folders, then delete the `data-bin/group_sample_seedXX.csv` file in `data-bin`. Then run `./scripts/a0_everything.ps1` to regenerate all reports and retrain all models.

# Streamlit App

The production Streamlit deployment doesn't include DistilBERT as running it crashes Streamlit.

Url: https://cse-3521-fall-26-team-beta-ai-food-reviews.streamlit.app/
