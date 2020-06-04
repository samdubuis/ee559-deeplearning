import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
# import seaborn as sns
from convnet import ConvNet
from convnet_dropout import ConvDropoutNet


def train_model(model, weight_sharing, aux_loss_lambda, optimizer, scheduler, early_stop_patience, X_train, y_train,
                X_val, y_val, mini_batch_size, epochs, criterion, verbose=0):
    """
    Train a model

    Parameters:
    model (Module): The model to train
    weight_sharing (boolean): Whether to use weight sharing
    aux_loss_lambda (float): The weight given to the auxiliary loss
    optimizer (Optimizer): The optimizer used to train the model
    scheduler (Scheduler): The scheduler used to adapt the learning rate
    early_stop_patience (int): The number of epochs we wait without improvement before stopping
    X_train (Tensor): The data used to train the model
    y_train (Tensor): The labels used to train the model
    X_val (Tensor): The data used to validate the model
    y_val (Tensor): The labels used to validate the model
    mini_batch_size (int): The size of each mini-batch
    epochs (int): The number of epochs
    criterion (Criterion): The criterion used to compute the losses
    versbose (int): Used to define the quantity of logs to print while training

    Returns:
    list(float): All the train losses
    list(float): All the validation losses
    """

    all_train_loss = []
    all_val_loss = []
    for e in range(epochs):
        model.train()
        train_loss = 0
        for b in range(0, X_train.size(0), mini_batch_size):
            output_target, output_class_l, output_class_r = model(X_train.narrow(0, b, mini_batch_size),
                                                                  weight_sharing)  # Forward pass
            loss_target = criterion(output_target,
                                    y_train.narrow(0, b, mini_batch_size)[:, 0])  # The loss wrt. the binary target
            loss_class_l = criterion(output_class_l,
                                     y_train.narrow(0, b, mini_batch_size)[:, 1])  # The loss wrt. the left image number
            loss_class_r = criterion(output_class_r, y_train.narrow(0, b, mini_batch_size)[:, 2])  # The loss wrt. the right image number

            # We compute the total loss (no auxiliary loss if aux_loss_lambda is 0)
            # If we set aux_loss_lambda to a very big value, it will basicaly shadow the target_loss (tried during validation)
            loss = loss_target + aux_loss_lambda * 0.5 * (loss_class_l + loss_class_r)

            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # For the validation, we change the mode of the model to eval()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b in range(0, X_val.size(0), mini_batch_size):
                output_target, output_class_l, output_class_r = model(X_val.narrow(0, b, mini_batch_size),
                                                                      weight_sharing)
                loss_target = criterion(output_target, y_val.narrow(0, b, mini_batch_size)[:, 0])
                loss_class_l = criterion(output_class_l, y_val.narrow(0, b, mini_batch_size)[:, 1])
                loss_class_r = criterion(output_class_r, y_val.narrow(0, b, mini_batch_size)[:, 2])

                loss = loss_target + aux_loss_lambda * 0.5 * (loss_class_l + loss_class_r)

                val_loss += loss.item()

            # If the validation loss has stoped improving, we lower the learning rate
            scheduler.step(val_loss)

            # We compute the mean loss by batch
            train_loss /= (X_train.size(0) // mini_batch_size)
            val_loss /= (X_val.size(0) // mini_batch_size)

            # We save all the training and validation losses
            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)

        if verbose == 1 and e % 10 == 0:
            print('{} {}'.format(e, train_loss, val_loss))

        # If we don't have improved the validation loss since a moment, we stop the training (to mitigate the overfitting)
        if len(all_val_loss) - (torch.Tensor(all_val_loss).argmin() + 1) > early_stop_patience:
            if verbose == 1:
                print('Early stopping')
            break

    return all_train_loss, all_val_loss


def split_train_test(X, y, train_size=0.8, shuffle=True):
    """
    Split data into training and testing sets

    Parameters:
    X (Tensor): The input data
    y (Tensor): The labels
    train_size (float): The fraction of data used for the training
    shuffle (boolean): Whether to randomly shuffle data before splitting

    Returns:
    Tensor: The training data
    Tensor: The training labels
    Tensor: The testing data
    Tensor: The testing labels
    """

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


# Compute the accuracy
def accuracy(y_pred, y_true):
    return 1 - (y_pred != y_true).to(torch.float32).mean().item()


# Build a specific model
def generate_model(model_name):
    if model_name == 'conv_net':
        return ConvNet()
    if model_name == 'conv_dropout_net':
        return ConvDropoutNet()
    # if model_name == 'conv_drop_batch_norm_net':
    #     return ConvDropBatchNormNet()
    raise Exception()


# Compute the number of trainable parameters of a model
def nb_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_multiple_times(model_name, weight_sharing, aux_loss_lambda, nb_iter, epochs, lr, X, y):
    """
    Do multiple trainings

    Parameters:
    model_name (String): The name of the model
    weight_sharing (boolean): Whether to use weight sharing
    aux_loss_lambda (float): The weight given to the auxiliary loss
    nb_iter (int): The number of trainings to do
    epochs (int): The number of epochs to train
    lr (float): The initial learning rate

    Returns:
    list(Model): The models we have trained
    list(list(float)): All the lists of train losses
    list(list(float)): All the lists of validation losses
    """

    models = []
    train_losses = []
    val_losses = []
    for i in range(nb_iter):
        mini_batch_size = 100

        # We define the criterion
        criterion = nn.CrossEntropyLoss()

        # We generate a new model
        model = generate_model(model_name)

        # We choose to use the Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # We define a scheduler to adapt the learning rate during training
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=7)
        # We define the number of epochs we wait without improvement before stopping the training
        early_stop_patience = 10

        # We split the training and validation data
        X_train, y_train, X_val, y_val = split_train_test(X, y)

        # We train the model
        train_loss, val_loss = train_model(model, weight_sharing, aux_loss_lambda, optimizer, scheduler,
                                           early_stop_patience, X_train, y_train, X_val, y_val, mini_batch_size, epochs,
                                           criterion, verbose=0)
        print('Iteration {}: Training loss={}, Validation loss={}'.format(i + 1, train_loss[-1], val_loss[-1]))

        models.append(model)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return models, train_losses, val_losses


# Plot all the training and validation losses from the multiple trainings
def plot_training(model_name, aux_loss_lambda, train_losses, val_losses):
    plt.figure()
    for i in range(len(train_losses)):
        plt.plot(train_losses[i], c='blue')
        plt.plot(val_losses[i], c='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss evolution (auxiliary loss lambda = {})'.format(aux_loss_lambda))


# Test all the models gathered from the multiple trainings
def plot_testing(model_name, weight_sharing, aux_loss_lambda, models, X, y, X_test, y_test):
    X_train, y_train = X, y

    all_train_class_acc = []
    all_train_target_acc = []
    all_test_class_acc = []
    all_test_target_acc = []
    for model in models:
        model.eval()  # We set the mode of the model to eval()
        target_train, class_l_train, class_r_train = model(X_train, weight_sharing)  # Get training predictions
        target_test, class_l_test, class_r_test = model(X_test, weight_sharing)  # Get testing predictions

        # We get the training and testing accuracy for the classes predictions (mean accuracy for left and right image)
        all_train_class_acc.append(
            0.5 * (accuracy(class_l_train.argmax(1), y_train[:, 1]) + accuracy(class_r_train.argmax(1), y_train[:, 2])))
        all_test_class_acc.append(
            0.5 * (accuracy(class_l_test.argmax(1), y_test[:, 1]) + accuracy(class_r_test.argmax(1), y_test[:, 2])))

        # We get the training and testing accuracy for the binary targets predictions
        all_train_target_acc.append(accuracy(target_train.argmax(1), y_train[:, 0]))
        all_test_target_acc.append(accuracy(target_test.argmax(1), y_test[:, 0]))

    # We compute the mean testing accuracy among all the trainings
    mean_target_acc = sum(all_test_target_acc) / len(all_test_target_acc)
    # And its standard error
    standard_error_target_acc = torch.Tensor(all_test_target_acc).std()

    plt.figure()
    # ax = sns.boxplot(data=[all_train_class_acc, all_train_target_acc, all_test_class_acc, all_test_target_acc])
    # ax.set_title('Accuracy on test = {} ± {} (auxiliary loss lambda = {})'.format('%.3f' % mean_target_acc, '%.3f' % standard_error_target_acc, aux_loss_lambda))
    # ax.set_xticklabels(['Training Class', 'Training Target', 'Test Class', 'Test Target'])
    # ax.set(ylabel='Accuracy')

    return mean_target_acc, standard_error_target_acc


# Run a full analysis of a model (multiple trainings + results visualization and comparison)
def run_full_analysis(model_name, weight_sharing, aux_loss_lambda, nb_iter, epochs, lr, X, y, X_test, y_test):
    # We train our model multiple times
    models, train_losses, val_losses = train_multiple_times(model_name, weight_sharing, aux_loss_lambda, nb_iter,
                                                            epochs, lr, X, y)
    # We plot all the training and validation losses curves
    plot_training(model_name, aux_loss_lambda, train_losses, val_losses)
    # We get the final accuracies
    acc, error = plot_testing(model_name, weight_sharing, aux_loss_lambda, models, X, y, X_test, y_test)

    return acc, error
