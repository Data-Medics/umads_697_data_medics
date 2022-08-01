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

# + tags=["parameters"]
upstream = []
nrows = None

# +
sys.path.insert(0, "..")

# project imports
import locations as loc

# run model on gpu if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if true, lemmatize the raw tweets
# if false, load the saved, processed tweets (assumes the lemmatized tweets already exist)
lemmatize_texts = False
# -
# location of saved lemmatized texts
lemma_sents_file_location = os.path.join(".", "output", "train_lemma_sents.csv")
dev_lemma_sents_file_location = os.path.join(".", "output", "dev_lemma_sents.csv")
test_lemma_sents_file_location = os.path.join(".", "output", "test_lemma_sents.csv")

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

df.head()

dev_df.head()

test_df.head()

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

# +
# view a view teets to make sure data is correct
# lemma_docs[:5]

# +
# #### Prepare and build the RNN

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


# +
# pytorch helper functions

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
from torch.utils.data import DataLoader

max_words = 50
train_batch_size = 512
val_batch_size = 1024
test_batch_size = 1024

# function will run on the batches of data BEFORE they are passed to the model
def collate(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_transform(_label))
        # t_text = text_transform(_text)[:max_words]
        processed_text = torch.tensor(text_transform(_text)[:max_words])
        text_list.append(processed_text)
    return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0).T

# instantiate the data loaders
train_loader = DataLoader(lemma_docs, batch_size=train_batch_size, collate_fn=collate, shuffle=True)
val_loader = DataLoader(dev_lemma_docs, batch_size=val_batch_size, collate_fn=collate)
test_loader = DataLoader(test_lemma_docs, batch_size=test_batch_size, collate_fn=collate)


# +
# define the the LSTM

from torch import nn
from torch.nn import functional as F

embedded_len = 64
hidden_dim = 128
n_layers=1

class TweetClassifier(nn.Module):
    def __init__(self):
        super(TweetClassifier, self).__init__()
        
        # create the embedding layer, will output vectors of embedded_len length
        self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedded_len)
        
        # create the lstm
        self.lstm = nn.LSTM(input_size=embedded_len, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        
        # feed the output of the lstm into a fully connected later
        self.fc_1 = nn.Linear(hidden_dim, 128)
        
        self.fc_2 = nn.Linear(128, len(label_encoder_dict))

    def forward(self, text):
        # get the embedded texts
        embeddings = self.embedding_layer(text)
        
        # build the hidden states
        hidden, carry = torch.randn(n_layers, len(text), hidden_dim), torch.randn(n_layers, len(text), hidden_dim)
        
        # get the output from the lstm
        output, (hidden, carry) = self.lstm(embeddings, (hidden, carry))
        
        output = self.fc_1(output[:,-1])
        output = self.fc_2(output)
        return output


# -

output_dim = len(label_encoder_dict)

# +
# view the model architecture
tweet_lstm_classifier = TweetClassifier()

tweet_lstm_classifier

# +
# check that all the model dimensions fit correctly
# and the output is of the desired shape
output_tensor = tweet_lstm_classifier(torch.randint(0, len(vocab), (1024, max_words)))

assert output_tensor.shape[0] == test_batch_size
assert output_tensor.shape[1] == output_dim

# +
# settings for model training run

from torch.optim import Adam

epochs = 25
learning_rate = 1e-3

criterion = nn.CrossEntropyLoss()
tweet_classifier = TweetClassifier()
optimizer = Adam(tweet_classifier.parameters(), lr=learning_rate)
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
    tweet_classifier.train()
    total_acc, total_count = 0, 0
    start_time = time.time()
    for idx, (label, text) in enumerate(train_loader):
        #         zero out the gradient for a new run
        optimizer.zero_grad()
        # create a prediction
        predicted_label = tweet_classifier(text)
        # calculate loss and run backprop
        # print(label)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tweet_classifier.parameters(), 0.1)
        # update weights
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % batch_log_freq == 0 and idx > 0:
            accuracy = total_acc / total_count
            print(f"Epoch: {epoch} Batch: {idx} of {total_batches_per_epoch}.  Accuracy: {accuracy:.2f}\n")
            total_acc, total_count = 0, 0
            start_time = time.time()
            
    tweet_classifier.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(val_loader):
            predicted_label = tweet_classifier(text)
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
        predicted_label = tweet_classifier(text)
        preds = predicted_label.argmax(1)
        full_preds.extend(preds)
y_pred = [a.item() for a in full_preds]

# +
from sklearn.metrics import accuracy_score, f1_score

y_true = [a[0] for a in test_lemma_docs]
accuracy_score(y_true, y_pred)
# -

f1_score(y_true, y_pred, average='macro')

# Notes and resources:
# * https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
# * https://coderzcolumn.com/tutorials/artificial-intelligence/pytorch-lstm-for-text-classification-tasks
# * https://github.com/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb

