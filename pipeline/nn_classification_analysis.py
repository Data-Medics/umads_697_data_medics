# ## Classification Comparison Using Neural Networks

import os
import pandas as pd
import sys
import spacy
from spacy.language import Language
import time
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score

upstream = []
nrows = None
epochs = None

# +
sys.path.insert(0, "..")

# project imports
import locations as loc

from nn_models import TweetClassificationLSTM, TweetClassificationEmbedder

# run model on gpu if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if true, lemmatize the raw tweets
# if false, load the saved, processed tweets (assumes the lemmatized tweets already exist)
lemmatize_texts = False
# -

# location of saved lemmatized texts
lemma_sents_file_location = os.path.join(loc.outputs, "train_lemma_sents.csv")
dev_lemma_sents_file_location = os.path.join(loc.outputs, "dev_lemma_sents.csv")
test_lemma_sents_file_location = os.path.join(loc.outputs, "test_lemma_sents.csv")

# load a spacy language model
nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words

# here are the transformations the spacy nlp object will perform on every doc
nlp.pipeline

# +
# load the data, sample the DF if interested
# mostly just to get the pipeline working, training/test should be done on the full data set
data_path = os.path.join(loc.data, "all_combined", "all_train.tsv")
dev_data_path = os.path.join(loc.data, "all_combined", "all_dev.tsv")
test_data_path = os.path.join(loc.data, "all_combined", "all_test.tsv")

if isinstance(nrows, int):
    df = pd.read_csv(data_path, sep="\t", nrows=nrows)
    dev_df = pd.read_csv(dev_data_path, sep="\t", nrows=nrows)
    test_df = pd.read_csv(test_data_path, sep="\t", nrows=nrows)
else:
    df = pd.read_csv(data_path, sep="\t")
    dev_df = pd.read_csv(dev_data_path, sep="\t")
    test_df = pd.read_csv(test_data_path, sep="\t")
# -

label_encoder_dict = {i: idx for idx, i in enumerate(df["class_label"].unique())}

# +
# apply the lemmatizer to al lthe tweets
# and save the outputs to csv files
if lemmatize_texts:
    # apply the spacy pipeline to the tweets
    docs = df["tweet_text"].apply(lambda x: nlp(x))
    labels = df["class_label"].apply(lambda a: label_encoder_dict[a])
    pd.DataFrame({"labels": labels, "tweet_text": docs}).to_csv(lemma_sents_file_location, index=False)

    # apply the spacy pipeline to the tweets
    dev_docs = dev_df["tweet_text"].apply(lambda x: nlp(x))
    dev_labels = dev_df["class_label"].apply(lambda a: label_encoder_dict[a])
    pd.DataFrame({"labels": dev_labels, "tweet_text": dev_docs}).to_csv(dev_lemma_sents_file_location, index=False)

    # apply the spacy pipeline to the tweets
    test_docs = test_df["tweet_text"].apply(lambda x: nlp(x))
    test_labels = test_df["class_label"].apply(lambda a: label_encoder_dict[a])
    pd.DataFrame({"labels": test_labels, "tweet_text": test_docs}).to_csv(test_lemma_sents_file_location, index=False)

else:
    print("'lemmatize_texts' is set to False - loading saved texts")

# +
# read the lemmatized sentence
t = pd.read_csv(lemma_sents_file_location)
lemma_docs = list(zip(t.labels.to_list(), t.tweet_text.to_list()))

t = pd.read_csv(dev_lemma_sents_file_location)
dev_lemma_docs = list(zip(t.labels.to_list(), t.tweet_text.to_list()))

t = pd.read_csv(test_lemma_sents_file_location)
test_lemma_docs = list(zip(t.labels.to_list(), t.tweet_text.to_list()))


# -

# #### Prepare Data and Build Helper Functions

# +
# pytorch helper functions
# used with `build_vocab_from_iterator` to build the word to token mapping

def yield_tokens(doc_strings):
    # discard the label because it does not need to be tokenized
    for _, text in doc_strings:
        # yield the tokenized text
        yield tokenizer(text)


# +
# use the torchtext tokenizer
# a little redundent because we have spacy, but this allows for the entire pipeline to run
# in torch if we want

tokenizer = get_tokenizer('basic_english')
# -

# build the torch encodings
# add a special character for out of bag words
vocab = build_vocab_from_iterator(yield_tokens(lemma_docs), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# convenience functions to transform data
text_transform = lambda x: vocab(tokenizer(x))
label_transform = lambda x: int(x)


# +
def collate_batch_embedder(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_transform(_label))
        processed_text = torch.tensor(text_transform(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

# function will run on the batches of data BEFORE they are passed to the model
def collate_batch_lstm(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_transform(_label))
        # t_text = text_transform(_text)[:max_words]
        processed_text = torch.tensor(text_transform(_text)[:max_words])
        text_list.append(processed_text)
    return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0).T


# -

# number of output layers
# should be 10 because there are 10 classes
output_dim = len(label_encoder_dict)

# ### Embedding Bag Model

# +
# model run variables
vocab_size = len(vocab)
emsize = 64
batch_log_freq = 25


# instantiate model
tweet_embedding_classifier = TweetClassificationEmbedder(vocab_size, emsize, output_dim).to(device)


# +
# define the training loop
def train(dataloader):
    tweet_embedding_classifier.train()
    total_acc, total_count = 0, 0

    for idx, (label, text, offsets) in enumerate(dataloader):
        # zero out the gradient for a new run
        optimizer.zero_grad()
        # create a prediction
        predicted_label = tweet_embedding_classifier(text, offsets)
        # calculate loss and run backprop
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tweet_embedding_classifier.parameters(), 0.1)
        # update weights
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % batch_log_freq == 0 and idx > 0:
            accuracy = total_acc / total_count
            print(f"Epoch: {epoch} Batch: {idx} of {total_batches_per_epoch}.  Accuracy: {accuracy:.2f}\n")
            total_acc, total_count = 0, 0


def evaluate(dataloader):
    tweet_embedding_classifier.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = tweet_embedding_classifier(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


# +
# Hyperparameters
# set from pipeline.yaml
# EPOCHS = 15
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(tweet_embedding_classifier.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None

train_loader = DataLoader(lemma_docs, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch_embedder)
val_loader = DataLoader(dev_lemma_docs, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch_embedder)
test_loader = DataLoader(test_lemma_docs, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch_embedder)

total_batches_per_epoch = len(train_loader)
# -

# model training loop
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_loader)
    accu_val = evaluate(val_loader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print('#' * 25)

    print(f"\nVALIDATION SET: Epoch: {epoch} Accuracy: {accu_val:.2f}\n")

    print('#' * 25)

# create predictions on the test set
full_preds = []
with torch.no_grad():
    for idx, (label, text, offsets) in enumerate(test_loader):
        predicted_label = tweet_embedding_classifier(text, offsets)
        preds = predicted_label.argmax(1)
        full_preds.extend(preds)
y_pred = [a.item() for a in full_preds]

# metrics on test set
y_true = [a[0] for a in test_lemma_docs]
accuracy_score(y_true, y_pred)

# more metrics on test set
f1_score(y_true, y_pred, average='macro')

# ### LSTM

# +
max_words = 50
train_batch_size = 512
val_batch_size = 1024
test_batch_size = 1024

# instantiate the data loaders
train_loader = DataLoader(lemma_docs, batch_size=train_batch_size, collate_fn=collate_batch_lstm, shuffle=True)
val_loader = DataLoader(dev_lemma_docs, batch_size=val_batch_size, collate_fn=collate_batch_lstm)
test_loader = DataLoader(test_lemma_docs, batch_size=test_batch_size, collate_fn=collate_batch_lstm)
# -

# neural net architecture settings
embedded_len = 64
hidden_dim = 128
n_layers=1

# +
# instantiate the the LSTM
tweet_lstm_classifier = TweetClassificationLSTM(vocab_size=vocab_size, 
                                                embedded_len=embedded_len, 
                                                hidden_dim=hidden_dim, 
                                                n_layers=n_layers, 
                                                output_len=output_dim)

# view the model architecture
tweet_lstm_classifier

# +
# check that all the model dimensions fit correctly
# and the output is of the desired shape
output_tensor = tweet_lstm_classifier(torch.randint(0, len(vocab), (1024, max_words)), n_layers, hidden_dim)

assert output_tensor.shape[0] == test_batch_size
assert output_tensor.shape[1] == output_dim

# +
# settings for model training run
# set from pipeline.yaml
# epochs = 25
learning_rate = 1e-3

criterion = nn.CrossEntropyLoss()
# tweet_classifier = TweetClassificationLSTM()
optimizer = Adam(tweet_lstm_classifier.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(tweet_classifier.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

total_accu = None
full_acc_score = 0
acc_score_counter = []
total_batches_per_epoch = len(train_loader)
batch_log_freq = 25

# +
# training loop - RUN AND EVALUATE THE MODEL

for epoch in range(1, epochs + 1):
    tweet_lstm_classifier.train()
    total_acc, total_count = 0, 0
    for idx, (label, text) in enumerate(train_loader):
        #         zero out the gradient for a new run
        optimizer.zero_grad()
        # create a prediction
        predicted_label = tweet_lstm_classifier(text, n_layers, hidden_dim)
        # calculate loss and run backprop
        # print(label)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tweet_lstm_classifier.parameters(), 0.1)
        # update weights
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % batch_log_freq == 0 and idx > 0:
            accuracy = total_acc / total_count
            print(f"Epoch: {epoch} Batch: {idx} of {total_batches_per_epoch}.  Accuracy: {accuracy:.2f}\n")
            total_acc, total_count = 0, 0
            
    tweet_lstm_classifier.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(val_loader):
            predicted_label = tweet_lstm_classifier(text, n_layers, hidden_dim)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    acc_score = total_acc / total_count
    
    if total_accu is not None and total_accu > acc_score:
        scheduler.step()
    else:
        total_accu = acc_score
    
    print(f"\nVALIDATION SET: Epoch: {epoch} Accuracy: {acc_score:.2f}\n")
    acc_improvement = acc_score - full_acc_score
    full_acc_score = acc_score
    if acc_improvement < 0.02:
        acc_score_counter.append(1)
    else:
        acc_score_counter.append(0)
    if len(acc_score_counter) > 5 and sum(acc_score_counter[-3:]) > 2:
        print("No validation accuracy improvement in last 3 epochs, terminating training loop")
# -

# create predictions on the test set
full_preds = []
with torch.no_grad():
    for idx, (label, text) in enumerate(test_loader):
        predicted_label = tweet_lstm_classifier(text, n_layers, hidden_dim)
        preds = predicted_label.argmax(1)
        full_preds.extend(preds)
y_pred = [a.item() for a in full_preds]

# metrics on test set
y_true = [a[0] for a in test_lemma_docs]
accuracy_score(y_true, y_pred)

# more metrics on test set
f1_score(y_true, y_pred, average='macro')

# Notes and resources:
# * https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
# * https://coderzcolumn.com/tutorials/artificial-intelligence/pytorch-lstm-for-text-classification-tasks
# * https://github.com/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb


