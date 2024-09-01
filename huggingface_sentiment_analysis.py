# -*- coding: utf-8 -*-
"""
HUGGINGFACE-SENTIMENT-ANALYSIS
"""

import pandas as pd
import numpy as np

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from torch.nn.functional import softmax

class ExtractSentiment:

  def __init__(self, checkpoint):

    self.checkpoint = checkpoint

    self.tokenizer = None
    self.classification_head = None

  # tokenization
  def create_tokenizer(self):

    tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

    self.tokenizer = tokenizer

  # extract model
  def create_model(self):

    model = AutoModelForSequenceClassification.from_pretrained(
        self.checkpoint)

    self.classification_head = model

  def train_model(self):

    self.create_tokenizer()

    self.create_model()

  def apply_softmax(self, output):

    output = softmax(output, dim=-1)

    return output


  def give_sentiment(self, raw_input):

    input = self.tokenizer(raw_input, padding=True
                           , truncation=True
                           , return_tensors="pt")

    output = self.classification_head(**input)

    output = self.apply_softmax(output.logits)

    output = output.detach().numpy()

    labels = self.classification_head.config.id2label

    res = pd.DataFrame(output, columns=labels.values())

    return res.idxmax(axis=1).values[0]

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

SentimentClassificationModel = ExtractSentiment(checkpoint)
SentimentClassificationModel.train_model()

sentence_to_check = "I hate the way you make me feel"
print(SentimentClassificationModel.give_sentiment([sentence_to_check]))

sentence_to_check = "I love the way you make me feel"
print(SentimentClassificationModel.give_sentiment([sentence_to_check]))



