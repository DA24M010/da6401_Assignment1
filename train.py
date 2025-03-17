import argparse
from nn import FeedforwardNN
import wandb
from optimizer import *

# -wp, --wandb_project	myprojectname	Project name used to track experiments in Weights & Biases dashboard
# -we, --wandb_entity	myname	Wandb Entity used to track experiments in the Weights & Biases dashboard.
# -d, --dataset	fashion_mnist	choices: ["mnist", "fashion_mnist"]
# -e, --epochs	1	Number of epochs to train neural network.
# -b, --batch_size	4	Batch size used to train neural network.
# -l, --loss	cross_entropy	choices: ["mean_squared_error", "cross_entropy"]
# -o, --optimizer	sgd	choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
# -lr, --learning_rate	0.1	Learning rate used to optimize model parameters
# -m, --momentum	0.5	Momentum used by momentum and nag optimizers.
# -beta, --beta	0.5	Beta used by rmsprop optimizer
# -beta1, --beta1	0.5	Beta1 used by adam and nadam optimizers.
# -beta2, --beta2	0.5	Beta2 used by adam and nadam optimizers.
# -eps, --epsilon	0.000001	Epsilon used by optimizers.
# -w_d, --weight_decay	.0	Weight decay used by optimizers.
# -w_i, --weight_init	random	choices: ["random", "Xavier"]
# -nhl, --num_layers	1	Number of hidden layers used in feedforward neural network.
# -sz, --hidden_size	4	Number of hidden neurons in a feedforward layer.
# -a, --activation	sigmoid	choices: ["identity", "sigmoid", "tanh", "ReLU"]

def train():
    pass

if(__name__ == '__main__'):
    
    parser = argparse.ArgumentParser(description="Train a neural network with various hyperparameters.")

    # Weights & Biases arguments
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401 Assignments", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", type=str, default="da24m010-indian-institute-of-technology-madras", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")

    # Dataset and training parameters
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset to use.")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to train the neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for training.")

    # Loss function and optimizer
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help="Loss function to use.")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="nadam", help="Optimizer to use.")

    # Learning rate and optimizer parameters
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate for optimization.")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="Momentum for momentum and nag optimizers.")
    parser.add_argument("-beta", "--beta", type=float, default=0.9, help="Beta for RMSprop optimizer.")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 for Adam and Nadam optimizers.")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 for Adam and Nadam optimizers.")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.00000001, help="Epsilon for optimizers.")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay for optimizers.")

    # Model architecture
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="random", help="Weight initialization method.")
    parser.add_argument("-nhl", "--num_layers", type=int, default=3, help="Number of hidden layers in the neural network.")
    parser.add_argument("-sz", "--hidden_size", type=int, default=64, help="Number of neurons in each hidden layer.")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="tanh", help="Activation function to use.")

    args = parser.parse_args()
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity) 
    run.name = f"LR_{args.learning_rate}_HL_{args.num_layers}_HLS_{args.hidden_size}_OPT_{args.optimizer}_ACTIVATION_{args.activation}_NUM_EPOCHS_{args.epochs}_BATCH_SIZE_{args.batch_size}_W_INIT_{args.weight_init}_W_DECAY_{args.weight_decay}_LOSS_{args.loss}"

    model = FeedforwardNN(num_layers=args.num_layers, hidden_size=args.hidden_size, learning_rate=args.learning_rate, momentum=args.momentum, 
                          activation=args.activation, optimizer=args.optimizer, loss_function=args.loss, weight_init=args.weight_init, 
                          beta=args.beta, epsilon=args.epsilon, beta1=args.beta1, beta2=args.beta2, weight_decay=args.weight_decay)

    model.train(epochs=args.epochs, batch_size=args.batch_size, dataset = args.dataset, wandb_logs= True)


