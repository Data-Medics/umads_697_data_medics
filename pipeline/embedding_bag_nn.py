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

# + tags=["parameters"]
upstream = []
nrows = None
# -

# project settings

# +
sys.path.insert(0, "..")

# project imports
import locations as loc

# run model on gpu if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -
# load a spacy language model
nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words

# here are the transformations the spacy nlp object will perform on every doc
nlp.pipeline

# +
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

# apply the spacy pipeline to the tweets
docs = df["tweet_text"].apply(lambda x: nlp(x))
labels = df["class_label"].apply(lambda a: label_encoder_dict[a])

# apply the spacy pipeline to the tweets
dev_docs = dev_df["tweet_text"].apply(lambda x: nlp(x))
dev_labels = dev_df["class_label"].apply(lambda a: label_encoder_dict[a])

# apply the spacy pipeline to the tweets
test_docs = df["tweet_text"].apply(lambda x: nlp(x))
test_labels = df["class_label"].apply(lambda a: label_encoder_dict[a])

# +

# create a corpus of lemmatized docs
lemma_docs = []
for label, doc in zip(labels, docs):
    # if we want to remove stopwords
    #     lemma_docs.append((label, ' '.join([token.lemma_ for token in doc if
    #                                         token.lemma_ not in stopwords])))
    # if we want to keep stopwords in the corpus
    lemma_docs.append((label, ' '.join([token.lemma_ for token in doc])))
# -

# create a corpus of lemmatized docs
dev_lemma_docs = []
for label, doc in zip(dev_labels, dev_docs):
    # if we want to remove stopwords
    #     lemma_docs.append((label, ' '.join([token.lemma_ for token in doc if
    #                                         token.lemma_ not in stopwords])))
    # if we want to keep stopwords in the corpus
    dev_lemma_docs.append((label, ' '.join([token.lemma_ for token in doc])))

# create a corpus of lemmatized docs
test_lemma_docs = []
for label, doc in zip(test_labels, test_docs):
    # if we want to remove stopwords
    #     lemma_docs.append((label, ' '.join([token.lemma_ for token in doc if
    #                                         token.lemma_ not in stopwords])))
    # if we want to keep stopwords in the corpus
    test_lemma_docs.append((label, ' '.join([token.lemma_ for token in doc])))

# +
# see the lemmatized docs
# lemma_docs
# -

lemma_sents_file_location = os.path.join(".", "output", "train_lemma_sents.txt")
dev_lemma_sents_file_location = os.path.join(".", "output", "dev_lemma_sents.txt")
test_lemma_sents_file_location = os.path.join(".", "output", "test_lemma_sents.txt")

# +
# checkpoint - write lemmatized sentences to txt

with open(lemma_sents_file_location, "w") as f:
    for s in lemma_docs:
        f.write(str(s[0]))
        f.write(s[1])
# -

with open(dev_lemma_sents_file_location, "w") as f:
    for s in dev_lemma_docs:
        f.write(str(s[0]))
        f.write(s[1])

with open(test_lemma_sents_file_location, "w") as f:
    for s in test_lemma_docs:
        f.write(str(s[0]))
        f.write(s[1])

# +
# read the lemmatized sentence
with open(lemma_sents_file_location, "r") as f:
    doc = f.readlines()

lemma_docs = []
for text in doc:
    lemma_docs.append((text[0], text[1:]))

# +
with open(dev_lemma_sents_file_location, "r") as f:
    doc = f.readlines()

dev_lemma_docs = []
for text in doc:
    dev_lemma_docs.append((text[0], text[1:]))

# +
with open(test_lemma_sents_file_location, "r") as f:
    doc = f.readlines()

test_lemma_docs = []
for text in doc:
    test_lemma_docs.append((text[0], text[1:]))

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


def collate_batch(batch):
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

# +
text_transform = lambda x: vocab(tokenizer(x))
label_transform = lambda x: int(x)


# create the data loader
# dataloader = DataLoader(lemma_docs, batch_size=8, shuffle=False, collate_fn=collate_batch)

# -

# define the model
class TweetClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TweetClassificationModel, self).__init__()
        # use an EmbeddingBag as the "text" portion of the model
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        # traditional Linear layer as the final output
        self.fc = nn.Linear(embed_dim, num_class)
        # initialize the weights - this will help convergence
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        # run a forward pass through the NN
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


# model run variables
num_class = len(set([a[0] for a in lemma_docs]))
vocab_size = len(vocab)
emsize = 64
# instantiate model
model = TweetClassificationModel(vocab_size, emsize, num_class).to(device)


# +
# define the training loop
def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        #         zero out the gradient for a new run
        optimizer.zero_grad()
        # create a prediction
        predicted_label = model(text, offsets)
        # calculate loss and run backprop
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        # update weights
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % 100 == 0 and idx > 0:
            accuracy = total_acc / total_count
            print(f"Epoch: {epoch} Batch: {idx / 100} Accuracy: {accuracy:.2f}\n")
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


# +
from torchtext.data.functional import to_map_style_dataset

# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None

train_dataloader = DataLoader(lemma_docs, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(dev_lemma_docs, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_lemma_docs, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print('#' * 25)

    print(f"\nVALIDATION SET: Epoch: {epoch} Accuracy: {accu_val:.2f}\n")

    print('#' * 25)