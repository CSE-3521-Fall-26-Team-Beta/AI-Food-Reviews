'Generates DistilBERT modern model. Run from the repository root with "python -m scripts.p6_distilbert" or "python -m scripts.p6_distilbert --epochs <epoch count (default 3)>" to train, and "python -m scripts.p6_distilbert --review <your review text in quotes>" to predict.'

import argparse
import re
from os import makedirs
from os.path import exists

import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DistilBertModel

from lib.constants import DISTILBERT_CHECKPOINT_PATH, DISTILBERT_REPORT_PATH, MODEL_DIR, SAMPLE_CSV_PATH, TRAINING_SEED
from lib.generate_group_sample import ensure_sample_csv_exists
from lib.ml_helpers import print_and_write

MODEL = 'distilbert/distilbert-base-uncased'
LABELS = ['negative', 'neutral', 'positive']

_device = None
_tokenizer = None
_model = None

class _SentimentModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = DistilBertModel.from_pretrained(MODEL)
    self.mlp = nn.Sequential(
      nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 3)
    )
  
  def forward(self, tokens):
    cls = self.encoder(**tokens).last_hidden_state[:, 0]
    return self.mlp(cls)

def _clean_text(text):
  return ' '.join(re.sub(r'<[^>]+>', ' ', text.lower()).split())

def _review_splits():
  ensure_sample_csv_exists()
  reviews = pd.read_csv(SAMPLE_CSV_PATH, usecols = ['Text', 'Score'])
  reviews['label'] = reviews['Score'].map({1: 0, 2: 0, 3: 1, 4: 2, 5: 2})
  reviews = reviews.dropna().copy()
  reviews['Text'] = reviews['Text'].map(_clean_text)
  consistent = reviews.groupby('Text')['label'].transform('nunique') == 1
  reviews = reviews[consistent & reviews['Text'].ne('')].drop_duplicates('Text')
  train, rest = train_test_split(
    reviews, test_size = 0.3, stratify = reviews.label, random_state = TRAINING_SEED
  )
  val, test = train_test_split(
    rest, test_size = 2 / 3, stratify = rest.label, random_state = TRAINING_SEED
  )
  return train, val, test

def _tokenize(texts):
  return _tokenizer(
    list(texts),
    padding = True,
    truncation = True,
    max_length = 256,
    return_tensors = 'pt'
  ).to(_device)

def _init_model_vars():
  global _device, _tokenizer, _model
  
  if _device == None:
    torch.manual_seed(TRAINING_SEED)
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _tokenizer = AutoTokenizer.from_pretrained(MODEL)
    _model = _SentimentModel().to(_device)

def _init_model_vars_with_saved_weights():
  if _device == None:
    _init_model_vars()
    _model.load_state_dict(torch.load(DISTILBERT_CHECKPOINT_PATH, map_location = _device, weights_only = True))
    _model.eval()

def train(epochs = 3):
  if not exists(DISTILBERT_REPORT_PATH):
    print('Training DistilBERT...')
    
    _init_model_vars()
    
    train, val, test = _review_splits()
    batches = DataLoader(
      list(zip(train.Text, train.label.astype(int))),
      batch_size = 16,
      shuffle = True,
    )
    optimizer = torch.optim.AdamW(_model.parameters(), lr = 2e-5)
    # Give neutral reviews twice the weight of negative or positive reviews.
    criterion = nn.CrossEntropyLoss(weight = torch.tensor([1., 2., 1.], device = _device))
    use_bfloat16 = _device.type == 'cuda' and torch.cuda.is_bf16_supported()
    
    @torch.no_grad()
    def evaluate(reviews):
      _model.eval()
      predictions = []
      for texts in DataLoader(reviews.Text.tolist(), batch_size = 16):
        predictions.extend(
          _model(_tokenize(texts))
            .argmax(dim = 1)
            .tolist()
        )
      return classification_report(
        reviews.label,
        predictions,
        labels = [0, 1, 2],
        target_names = LABELS,
        zero_division = 0,
        output_dict = True
      )
    
    makedirs(MODEL_DIR, exist_ok = True)
    
    with open(DISTILBERT_REPORT_PATH, 'w', newline = '') as f:
      best_f1 = -1
      for epoch in range(epochs):
        _model.train()
        for step, (texts, labels) in enumerate(batches, 1):
          optimizer.zero_grad()
          with torch.autocast(_device.type, dtype = torch.bfloat16, enabled = use_bfloat16):
            loss = criterion(_model(_tokenize(texts)), labels.to(_device))
          loss.backward()
          optimizer.step()
          if step % 250 == 0:
            print_and_write(f'Epoch {epoch + 1}, batch {step}/{len(batches)}', flush = True, file = f)
        f1 = evaluate(val)['macro avg']['f1-score']
        print_and_write(f'Epoch {epoch + 1} validation macro F1: {f1:.3f}', flush = True, file = f)
        if f1 > best_f1:
          torch.save(_model.state_dict(), DISTILBERT_CHECKPOINT_PATH)
          best_f1 = f1
      
      _model.load_state_dict(torch.load(DISTILBERT_CHECKPOINT_PATH, map_location = _device, weights_only = True))
      metrics = evaluate(test)
      print_and_write(f'Test accuracy: {metrics.pop('accuracy'):.3f}', file = f)
      print_and_write(pd.DataFrame(metrics).T.round(3), file = f)
    
    print(f'DistilBERT training report saved to {DISTILBERT_REPORT_PATH}')
  else:
    print('DistilBERT already trained')

def categorize(review_text, print_func = print):
  _init_model_vars_with_saved_weights()
  
  with torch.no_grad():
    probabilities = (
      _model(
        _tokenize([_clean_text(review_text)]),
      )
        .softmax(dim = 1)[0]
        .tolist()
    )
  
  prediction = LABELS[probabilities.index(max(probabilities))]
  prediction_probabilities = zip(LABELS, probabilities)
  
  if print_func != None:
    print_func(f'Prediction: {prediction}')
    
    print_func(
      'Prediction Probabilities:\n' + 
      '\n'.join(f'{label}: {probability:.1%}' for label, probability in prediction_probabilities)
    )
  else:
    return {
      'prediction': prediction,
      'probabilities': dict(prediction_probabilities),
    }

def main():
  parser = argparse.ArgumentParser(description = __doc__)
  parser.add_argument('--review', help = 'Classify a review with the saved model')
  parser.add_argument('--epochs', type = int, default = 3)
  args = parser.parse_args()
  if args.epochs < 1:
    parser.error('--epochs must be positive')
  
  if args.review is not None:
    categorize(args.review)
  else:
    train(args.epochs)

if __name__ == '__main__':
  main()
