import torch
import torch.nn as nn


class Train:

    def get_batches(self, arr_x, arr_y, batch_size):

        # iterate through the arrays
        prv = 0
        for n in range(batch_size, arr_x.shape[0], batch_size):
            # print(arr_x)
            x = arr_x[prv:n]
            y = arr_y[prv:n]
            prv = n
            yield x, y

    def train(self, x_int1, y_int1, net, epochs=1, batch_size=32, lr=0.001, clip=1, print_every=32):

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
            for x, y in self.get_batches(x_int1, y_int1, batch_size):
                counter += 1

                # convert numpy arrays to PyTorch arrays
                # inputs, targets = torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
                inputs, targets = x, y

                # push tensors to GPU
                # inputs, targets = inputs.cuda(), targets.cuda()
                inputs = inputs.type(torch.LongTensor)
                targets = targets.type(torch.LongTensor)
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
                    print("Epoch: {}/{}...".format(e + 1, epochs),
                          "Step: {}...".format(counter))
