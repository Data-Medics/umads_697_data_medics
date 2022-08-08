import torch
from torch import nn
from torch.nn import functional as F


class TweetClassificationLSTM(nn.Module):
    def __init__(self, vocab_size, embedded_len, hidden_dim, n_layers, output_len):
        super(TweetClassificationLSTM, self).__init__()
        
        # create the embedding layer, will output vectors of embedded_len length
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedded_len)
        
        # create the lstm
        self.lstm = nn.LSTM(input_size=embedded_len, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        
        # feed the output of the lstm into a fully connected later
        self.fc_1 = nn.Linear(hidden_dim, 128)
        
        self.fc_2 = nn.Linear(128, output_len)

    def forward(self, text, n_layers, hidden_dim):
        # get the embedded texts
        embeddings = self.embedding_layer(text)
        
        # build the hidden states
        hidden, carry = torch.randn(n_layers, len(text), hidden_dim), torch.randn(n_layers, len(text), hidden_dim)
        
        # get the output from the lstm
        output, (hidden, carry) = self.lstm(embeddings, (hidden, carry))
        
        output = self.fc_1(output[:,-1])
        output = self.fc_2(output)
        return output

    
class TweetClassificationEmbedder(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TweetClassificationEmbedder, self).__init__()
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