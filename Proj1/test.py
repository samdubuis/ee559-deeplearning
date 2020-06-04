import torch
import matplotlib.pyplot as plt
from dlc_practical_prologue import generate_pair_sets
from helper import nb_trainable_parameters, generate_model, run_full_analysis

plt.rcParams.update({'figure.max_open_warning': 0})


def main():
    # We set the seed for reproducibility
    torch.manual_seed(42)

    N = 1000
    # We generate the data
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(N)
    # We normalize the gray values
    train_input /= 255
    test_input /= 255

    # We define the input data X and the labels y
    X = train_input
    y = torch.cat((train_target.view(-1, 1), train_classes), 1)

    # Same for the test data
    X_test = test_input
    y_test = torch.cat((test_target.view(-1, 1), test_classes), 1)

    X.size(), y.size(), X_test.size(), y_test.size()

    # We plot some pairs for visualization
    fig, ax = plt.subplots(5, 2, figsize=(5, 15))
    for i, (img1, img2) in enumerate(train_input[:5]):
        ax[i, 0].imshow(img1, cmap='gray')
        ax[i, 1].imshow(img2, cmap='gray')

    # We try multiple auxiliary loss lambdas to maximize the target prediction accuracy without weight sharing
    model_name = 'conv_net'
    print('Number of trainable parameters:', nb_trainable_parameters(generate_model(model_name)))

    accs = []
    errors = []
    aux_loss_lambdas = [0, 0.1, 0.5, 1, 3, 10, 100, 1000]
    for aux_loss_lambda in aux_loss_lambdas:
        print('Auxiliary loss lambda:', aux_loss_lambda)
        acc, error = run_full_analysis(model_name, weight_sharing=False, aux_loss_lambda=aux_loss_lambda, nb_iter=10,
                                       epochs=80, lr=1e-3, X=X, y=y, X_test=X_test, y_test=y_test)
        print('Test accuracy: {} ± {}'.format('%.3f' % acc, '%.3f' % error))
        accs.append(acc)
        errors.append(error)

    plt.figure()
    plt.errorbar(aux_loss_lambdas, accs, yerr=errors)
    plt.xscale('symlog')
    plt.xlabel('Auxiliary loss lambda')
    plt.ylabel('Test target accuracy')
    plt.title('Test accuracy for different auxiliary loss lambdas')

    # We try multiple auxiliary loss lambdas to maximize the target prediction accuracy with weight sharing
    model_name = 'conv_net'
    print('Number of trainable parameters:', nb_trainable_parameters(generate_model(model_name)))

    accs = []
    errors = []
    aux_loss_lambdas = [0, 0.1, 0.5, 1, 3, 10, 100, 1000]
    for aux_loss_lambda in aux_loss_lambdas:
        print('Auxiliary loss lambda:', aux_loss_lambda)
        acc, error = run_full_analysis(model_name, weight_sharing=True, aux_loss_lambda=aux_loss_lambda, nb_iter=10,
                                       epochs=80, lr=1e-3, X=X, y=y, X_test=X_test, y_test=y_test)
        print('Test accuracy: {} ± {}'.format('%.3f' % acc, '%.3f' % error))
        accs.append(acc)
        errors.append(error)

    plt.figure()
    plt.errorbar(aux_loss_lambdas, accs, yerr=errors)
    plt.xscale('symlog')
    plt.xlabel('Auxiliary loss lambda')
    plt.ylabel('Test target accuracy')
    plt.title('Test accuracy for different auxiliary loss lambdas')

    # Run final analysis with best auxiliary loss lambda
    # One time with weight sharing
    model_name = 'conv_net'
    acc, error = run_full_analysis(model_name, weight_sharing=True, aux_loss_lambda=10, nb_iter=20, epochs=80, lr=1e-3,
                                   X=X, y=y, X_test=X_test, y_test=y_test)
    print('Test accuracy: {} ± {}'.format('%.3f' % acc, '%.3f' % error))

    # One time without weight sharing
    model_name = 'conv_net'
    acc, error = run_full_analysis(model_name, weight_sharing=False, aux_loss_lambda=10, nb_iter=20, epochs=80, lr=1e-3,
                                   X=X, y=y, X_test=X_test, y_test=y_test)
    print('Test accuracy: {} ± {}'.format('%.3f' % acc, '%.3f' % error))

    # We try multiple auxiliary loss lambdas to maximize the target prediction accuracy with weight sharing and dropout
    model_name = 'conv_dropout_net'
    print('Number of trainable parameters:', nb_trainable_parameters(generate_model(model_name)))

    accs = []
    errors = []
    aux_loss_lambdas = [0, 0.1, 0.5, 1, 3, 10, 100, 1000]
    for aux_loss_lambda in aux_loss_lambdas:
        print('Auxiliary loss lambda:', aux_loss_lambda)
        # Learning rate can be increased when we use dropout
        acc, error = run_full_analysis(model_name, weight_sharing=True, aux_loss_lambda=aux_loss_lambda, nb_iter=10,
                                       epochs=80, lr=1e-2, X=X, y=y, X_test=X_test, y_test=y_test)
        print('Test accuracy: {} ± {}'.format('%.3f' % acc, '%.3f' % error))
        accs.append(acc)
        errors.append(error)

    plt.figure()
    plt.errorbar(aux_loss_lambdas, accs, yerr=errors)
    plt.xscale('symlog')
    plt.xlabel('Auxiliary loss lambda')
    plt.ylabel('Test target accuracy')
    plt.title('Test accuracy for different auxiliary loss lambdas')

    # Run final analysis with best auxiliary loss lambda
    # One time with weight sharing
    model_name = 'conv_dropout_net'
    acc, error = run_full_analysis(model_name, weight_sharing=True, aux_loss_lambda=1, nb_iter=20, epochs=80, lr=1e-2,
                                   X=X, y=y, X_test=X_test, y_test=y_test)
    print('Test accuracy: {} ± {}'.format('%.3f' % acc, '%.3f' % error))

    # One time without weight sharing
    model_name = 'conv_dropout_net'
    acc, error = run_full_analysis(model_name, weight_sharing=False, aux_loss_lambda=1, nb_iter=20, epochs=80, lr=1e-2,
                                   X=X, y=y, X_test=X_test, y_test=y_test)
    print('Test accuracy: {} ± {}'.format('%.3f' % acc, '%.3f' % error))


if __name__ == "__main__":
    main()
