from simpleTorch import *
import matplotlib.pyplot as plt  # only used for plotting

torch.manual_seed(0)
torch.set_grad_enabled(False)

# generate the data
N = 1000
X_train, y_train = generate_data(N)
X_test, y_test = generate_data(N)

fig = plt.figure(figsize=(6, 6))
for i, x in enumerate(X_train):
    if y_train[i] == 1:
        plt.plot(x[0], x[1], 'b.')
    else:
        plt.plot(x[0], x[1], 'r.')

plt.savefig("data.png")

# normalize the features
X_train = normalize(X_train)
X_test = normalize(X_test)

# one-hot encode the labels
y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

# seperate the training set to get a validation set
X_train, y_train, X_val, y_val = split_train_test(X_train, y_train)

# function to create the linear model asked for the project
def create_linear_model(input_dim, nb_hidden, output_dim):
    return Sequential([
        Linear(input_dim, nb_hidden),
        Tanh(),
        Linear(nb_hidden, nb_hidden),
        Tanh(),
        Linear(nb_hidden, nb_hidden),
        Tanh(),
        Linear(nb_hidden, output_dim),
        Sigmoid()])


input_dim = 2
nb_hidden = 25
output_dim = 2

epochs = 1000
batch_size = 10

lr = 0.1
reduce_lr_patience = 10
reduce_lr_factor = 0.2
early_stop_patience = 100

model = create_linear_model(input_dim, nb_hidden, output_dim)
optimizer = SGD(model.param(), lr, reduce_lr_patience, reduce_lr_factor, early_stop_patience)

train_loss, train_acc, val_loss, val_acc = train_model(
    model, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size)

fig, ax = plt.subplots(1, 2, figsize=(15, 10))

ax[0].plot(train_loss)
ax[0].plot(val_loss)
ax[0].legend(['Training', 'Validation'])
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].set_title('Loss evolution')

ax[1].plot(train_acc)
ax[1].plot(val_acc)
ax[1].legend(['Training', 'Validation'])
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].set_title('Accuracy evolution')

plt.savefig("loss-acc.png")

result = test_model(model, X_test, y_test, batch_size)
print("Final test accuracy is {}" .format(result))
