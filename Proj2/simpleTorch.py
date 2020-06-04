import torch
import math


# definition of the general Module class
class Module(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


# definition of the general Linear class
class Linear(Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input = 0
        # init the weight tensor
        self.weight = torch.Tensor(input_dim, output_dim).normal_()
        # init the bias tensor
        self.bias = torch.Tensor(1, output_dim).normal_()

        # init the derivative tensors
        self.dl_dw = torch.Tensor(self.weight.size())
        self.dl_db = torch.Tensor(self.bias.size())

    def forward(self, input):
        # store the input for the backward step
        self.input = input
        output = self.input.mm(self.weight) + self.bias
        return output

    def backward(self, grdwrtoutput):
        self.dl_dw += self.input.t().mm(grdwrtoutput)
        self.dl_db += grdwrtoutput.mean(0) * self.input.size(0)

        output = grdwrtoutput.mm(self.weight.t())
        return output

    def param(self):
        # store the pairs of weights and derivatives
        return [(self.weight, self.dl_dw), (self.bias, self.dl_db)]


# definition of the general ReLu class : max(0, x)
class ReLu(Module):
    def __init__(self):
        super().__init__()
        self.s = 0

    def forward(self, input):
        self.s = input
        return input.clamp(min=0.0)

    def backward(self, grdwrtoutput):
        drelu = self.s.sign().clamp(min=0.0)
        return grdwrtoutput * drelu

    def param(self):
        return []


# definition of the general LeakyReLu class : max(alpha*x, x)
class LeakyReLu(Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.s = 0
        self.alpha = alpha

    def forward(self, input):
        self.s = input
        return input.clamp(min=0.0) + self.alpha*input.clamp(max=0.0)

    def backward(self, grdwrtoutput):
        drelu = torch.ones(self.s.size())
        drelu[self.s < 0] = self.alpha
        return grdwrtoutput * drelu

    def param(self):
        return []


# definition of the general tanh class
class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.s = 0

    def forward(self, input):
        self.s = input
        return input.tanh()  # call the func

    def backward(self, grdwrtoutput):
        dtanh = 1 - self.s.tanh().pow(2)  # formula of deriv of tanh
        return grdwrtoutput * dtanh

    def param(self):
        return []


# definition of the general sigmoid class
class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.s = 0

    def forward(self, input):
        self.s = input
        return self.sigmoid_f(input)  # call the sigmoid func below

    def backward(self, grdwrtoutput):
        dsigmoid = self.sigmoid_f(self.s) * (1 - self.sigmoid_f(self.s))
        return grdwrtoutput * dsigmoid

    def sigmoid_f(self, x):
        return 1 / (1 + torch.exp(-x))

    def param(self):
        return []


# definition of the general Sequential class
class Sequential(Module):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def add_module(self, ind, module):
        # add the module to the list of modules
        self.modules.append(module)
        return module

    def forward(self, input):
        output = input
        for module in self.modules:
            # apply forward of each module to the input
            output = module.forward(output)
        return output

    def backward(self, grdwrtoutput):
        output = grdwrtoutput
        for module in self.modules[::-1]:
            # apply backward of each module in reverse order
            output = module.backward(output)

    def param(self):
        parameters = []
        for module in self.modules:
            # append all the parameters of all the modules
            parameters.append(module.param())
        return parameters


# definition of the general SGD class
class SGD():
    def __init__(self, params, lr, reduce_lr_patience, reduce_lr_factor, early_stop_patience, monitor='val'):
        self.params = params  # the parameters of the model
        self.lr = lr  # the learning rate
        self.plateau_counter = 0  # the counter to know since how many epochs we are stucked in a local minima
        self.reduce_lr_patience = reduce_lr_patience  # the number of epochs to wait stucked before reducing the learning rate
        self.reduce_lr_factor = reduce_lr_factor  # the factor by which we reduce the learning rate
        self.early_stop_patience = early_stop_patience  # the number of epochs to wait stucked before stopping the learning
        self.monitor = monitor  # the loss to monitor (validation or training)

    # perform the gradient descent step
    def step(self):
        for module in self.params:
            for weight, grad in module:
                # remove from weight learningrate*grad for each module in each param (perform gradient descent)
                weight -= self.lr * grad

    # reset the gradients to zero
    def zero_grad(self):
        for module in self.params:
            for weight, grad in module:
                grad.zero_()

    # reduce the learning rate based on the monitored loss
    def reduce_lr_on_plateau(self, loss):
        # if the feature is enabled
        if self.reduce_lr_patience is not None:
            self.plateau_counter += 1
            # if the last value of val_loss is equal to the min, then reset the counter
            if loss[-1] == min(loss):
                self.plateau_counter = 0
            # if counter bigger than the patience, reset and mul learning rate by reducing factor
            elif self.plateau_counter > self.reduce_lr_patience:
                self.plateau_counter = 0
                self.lr *= self.reduce_lr_factor
                print('New lr:', self.lr)

    # stop the training based on the monitored loss
    def early_stopping(self, loss):
        # if the feature is enabled
        if self.early_stop_patience is None:
            return False
        return torch.Tensor(loss).argmin() < len(loss) - self.early_stop_patience


# definition of the mean squared loss
class LossMSE(Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y):
        return 0.5 * (y_pred - y.float()).pow(2).mean(1).sum()

    def backward(self, y_pred, y):
        return (y_pred - y.float()) / y.size(1)

    def param(self):
        return []


# function for the training of the model
def train_model(model, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size):
    all_train_loss = []
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []

    for epoch in range(epochs):

        # Training -------------------------------------------------------------------------------

        train_loss = 0
        train_errors = 0
        for b in range(0, X_train.size(0), batch_size):
            # begin by setting all grad of the optimizer to 0
            optimizer.zero_grad()

            x = X_train[b:b+batch_size]
            y = y_train[b:b+batch_size]

            # will call forward of all modules of the model (Sequential)
            output = model.forward(x)

            # number of errors on training set
            train_errors += (output.argmax(1) != y.argmax(1)).sum()

            # compute the loss and its derivatives
            train_loss += LossMSE().forward(output, y.float())
            dl_dloss = LossMSE().backward(output, y.float())

            # will call backward of all modules of the model (Sequential)
            model.backward(dl_dloss)

            # perform the optimization step (gradient descent)
            optimizer.step()

        # store the training loss and accuracy
        train_loss = train_loss.item() / X_train.size(0)
        all_train_loss.append(train_loss)
        all_train_acc.append(1 - float(train_errors) / X_train.size(0))

        # Validation --------------------------------------------------------------------------------

        val_loss = 0
        val_errors = 0
        for b in range(0, X_val.size(0), batch_size):

            x = X_val[b:b+batch_size]
            y = y_val[b:b+batch_size]

            # will call forward of all modules of the model (Sequential)
            output = model.forward(x)

            # number of errors on the validation set
            val_errors += (output.argmax(1) != y.argmax(1)).sum()

            # compute the validation loss
            val_loss += LossMSE().forward(output, y.float())

        # store the validation loss and accuracy
        val_loss = val_loss.item() / X_val.size(0)
        all_val_loss.append(val_loss)
        all_val_acc.append(1 - float(val_errors) / X_val.size(0))

        if epoch % (epochs//20) == 0:
            print('Epoch: {}: train -> {:.5f}, validation -> {:.5f}'.format(epoch, train_loss, val_loss))

        # base on the loss to monitor, reduce learning size or stop earlier if needed
        loss_to_analyse = all_val_loss if optimizer.monitor == 'val' else all_train_loss
        optimizer.reduce_lr_on_plateau(loss_to_analyse)
        if optimizer.early_stopping(loss_to_analyse):
            print('Early Stopping')
            break

    return all_train_loss, all_train_acc, all_val_loss, all_val_acc


# function for testing the model
def test_model(model, X_test, y_test, batch_size):
    test_errors = 0
    for b in range(0, X_test.size(0), batch_size):
        x = X_test[b:b+batch_size]
        y = y_test[b:b+batch_size]

        # we compute the output by forwarding in all modules
        output = model.forward(x)

        # number of errors for this batch
        test_errors += (output.argmax(1) != y.argmax(1)).sum()

    test_acc = 1 - float(test_errors)/X_test.size(0)
    return test_acc


# function to generate n data uniformly in [0,1]² X, and labels Y
# depending of position outside (0) of disk of radius 1/sqrt(2*pi) else val 1
def generate_data(n):
    X = torch.empty(n, 2).uniform_()
    y = (((X[:, 0]-0.5).pow(2) + (X[:, 1]-0.5).pow(2)) <= 1 /
         (2*math.pi)).long().view(-1, 1)  # circle centered at 0.5, 0.5
    return X, y


# function to normalize data
def normalize(data):
    return (data - data.mean(0)) / data.std(0)


# function that splits dataset given in two, with percentage
# and possibility of shuffling
def split_train_test(X, y, train_size=0.8, shuffle=True):
    if shuffle:
        perm = torch.randperm(X.size(0))
        X = X[perm]
        y = y[perm]

    cut = int(train_size * X.size(0))
    X_train = X[:cut]
    y_train = y[:cut]
    X_val = X[cut:]
    y_val = y[cut:]

    return X_train, y_train, X_val, y_val


# function that takes labels and one hot encode them
def one_hot_encode(y):
    one_hot = torch.empty(y.size(0), 2).zero_()
    one_hot[torch.arange(y.size(0)), y[:, 0]] = 1
    return one_hot
