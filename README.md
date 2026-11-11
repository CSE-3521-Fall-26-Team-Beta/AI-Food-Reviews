# AI-Food-Reviews

This is the GitHub repository for the AI food review project for CSE 3521.

# Team Members

Ethan Boes, Max Wuchter, Haodong Qiu, Santosh Schaefer, Sulman Abdullahi

# Project Description

Our project will use the Amazon Fine Food Reviews dataset to build and evaluate a sentiment analysis system. We plan to use a shared sample of 50,000 reviews and convert the original star ratings into three sentiment categories: negative (1–2 stars), neutral (3 stars), and positive (4–5 stars).

We will preprocess the review text and apply TF-IDF and word embeddings as our features engineering methods to convert textual information into numerical representations. Next, we will implement two classical machine learning models, Naive Bayes and Gaussian Mixture Model (GMM), and compare their performance with a pre-trained transformer model, DistilBERT. We will evaluate the models using accuracy, precision, recall and F1-score.

Finally, we will develop and deploy a Streamlit web application that allows users to enter their own review text and view sentiment predictions from at least two trained models. Through this project, we aim to understand how different feature representations and model architectures influence the accuracy and efficiency of sentiment classification.
