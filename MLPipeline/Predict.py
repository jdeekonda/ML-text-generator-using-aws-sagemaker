import torch
import random
import torch.nn
import numpy as np
import torch.nn.functional as F

class Predict:
    # predict next token
    def predict(self, net, tkn, h=None):

        # tensor inputs
        from Engine import token2int
        x = np.array([[token2int[tkn]]])
        inputs = torch.from_numpy(x)

        # push to GPU
        # inputs = inputs.cuda()

        # detach hidden state from history
        h = tuple([each.data for each in h])

        # get the output of the model
        out, h = net(inputs, h)

        # get the token probabilities
        p = F.softmax(out, dim=1).data

        p = p.cpu()

        p = p.numpy()
        p = p.reshape(p.shape[1], )

        # get indices of top 3 values
        top_n_idx = p.argsort()[-3:][::-1]

        # randomly select one of the three indices
        sampled_token_index = top_n_idx[random.sample([0, 1, 2], 1)[0]]

        # return the encoded value of the predicted char and the hidden state
        from Engine import int2token
        return int2token[sampled_token_index], h

    # function to generate text
    def sample(self, net, size, prime='it is'):

        # push to GPU
        # net.cuda()
        net.eval()

        # batch size is 1
        h = net.init_hidden(1)

        toks = prime.split()

        # predict next token
        for t in prime.split():
            token, h = self.predict(net, t, h)
        toks.append(token)

        # predict subsequent tokens
        for i in range(size - 1):
            token, h = self.predict(net, toks[-1], h)
            toks.append(token)

        return ' '.join(toks)