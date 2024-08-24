
# Importing the Modules:
import subprocess
import sys
import os
import boto3
# Installing all the Pack

# subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "s3fs"])



import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import boto3
import argparse
import os
import random
import io

s3 = boto3.resource('s3')


obj1 = s3.Object('textgenerationbucket','inputs/int2token.json')
obj2 = s3.Object('textgenerationbucket','inputs/token2int.json')
int2token = json.load(obj1.get()['Body']) 
token2int = json.load(obj2.get()['Body'])

vocab_size = len(int2token)
print(vocab_size)


#Getting Inputs From S3:

import numpy as np

key1 = 'torch_data/x_int.npy'
key2 = 'torch_data/y_int.npy'
bucket = 'textgenerationbucket'

#Getting X AND Y :

x_int = np.load(io.BytesIO(s3.Object(bucket,key1).get()["Body"].read()))
y_int = np.load(io.BytesIO(s3.Object(bucket,key2).get()["Body"].read()))


# convert lists to numpy arrays
x_int = torch.tensor(np.array(x_int))
y_int = torch.tensor(np.array(y_int))
print(x_int[0])
print(y_int[0])

import argparse
import os


if __name__ =='__main__':
    
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Data, model, and output directories
#     parser.add_argument('model_dir', type=str, default=)
#     args, _ = parser.parse_known_args()
    model_dir  = os.environ['SM_MODEL_DIR']
    print(model_dir)

    class WordLSTM(nn.Module):

        def __init__(self, n_hidden=256, n_layers=4, drop_prob=0.3, lr=0.001):
            super().__init__()

            self.drop_prob = drop_prob
            self.n_layers = n_layers
            self.n_hidden = n_hidden
            self.lr = lr

            self.emb_layer = nn.Embedding(vocab_size, 200)

            ## define the LSTM
            self.lstm = nn.LSTM(200, n_hidden, n_layers, 
                                dropout=drop_prob, batch_first=True)

            ## define a dropout layer
            self.dropout = nn.Dropout(drop_prob)

            ## define the fully-connected layer
            self.fc = nn.Linear(n_hidden, vocab_size)      

        def forward(self, x, hidden):
            ''' Forward pass through the network. 
                These inputs are x, and the hidden/cell state `hidden`. '''

            ## pass input through embedding layer
            embedded = self.emb_layer(x)     

            ## Get the outputs and the new hidden state from the lstm
            lstm_output, hidden = self.lstm(embedded, hidden)

            ## pass through a dropout layer
            out = self.dropout(lstm_output)

            #out = out.contiguous().view(-1, self.n_hidden) 
            out = out.reshape(-1, self.n_hidden) 

            ## put "out" through the fully-connected layer
            out = self.fc(out)

            # return the final output and the hidden state
            return out, hidden


        def init_hidden(self, batch_size):
            ''' initializes hidden state '''
            # Create two new tensors with sizes n_layers x batch_size x n_hidden,
            # initialized to zero, for hidden state and cell state of LSTM
            weight = next(self.parameters()).data

            # if GPU is available
            if (torch.cuda.is_available()):
                hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())

            # if GPU is not available
            else:
                hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

            return hidden

    # instantiate the model
    net = WordLSTM()

    # push the model to GPU (avoid it if you are not using the GPU)
    net.cuda()

    print(net)

    def train(net, epochs=10, batch_size=32, lr=0.001, clip=1, print_every=32):

        # optimizer
        opt = torch.optim.Adam(net.parameters(), lr=lr)

        # loss
        criterion = nn.CrossEntropyLoss()

        # push model to GPU
        # net.cuda()

        counter = 0

        net.train()

        for e in range(epochs):

            # initialize hidden state
            h = net.init_hidden(batch_size)

            for x, y in get_batches(x_int, y_int, batch_size):

                counter+= 1

                # convert numpy arrays to PyTorch arrays
                # inputs, targets = torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
                inputs, targets = x, y

                # push tensors to GPU
                inputs, targets = inputs.cuda(), targets.cuda()

                # detach hidden states
                h = tuple([each.data for each in h])

                # zero accumulated gradients
                net.zero_grad()

                # get the output from the model
                output, h = net(inputs, h)

                # calculate the loss and perform backprop
                loss = criterion(output, targets.view(-1))

                # back-propagate error
                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(net.parameters(), clip)

                # update weigths
                opt.step()            

                if counter % print_every == 0:

                  print("Epoch: {}/{}...".format(e+1, epochs),
                        "Step: {}...".format(counter))


    def get_batches(arr_x, arr_y, batch_size):

        # iterate through the arrays
        prv = 0
        for n in range(batch_size, arr_x.shape[0], batch_size):
            x = arr_x[prv:n]
            y = arr_y[prv:n]
            prv = n
            yield x, y

    train(net, batch_size = 100, epochs=20, print_every=512)

    path = os.path.join(model_dir, "model.pt")
    torch.save(net.cpu().state_dict(), path)








