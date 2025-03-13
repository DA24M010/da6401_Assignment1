import wandb
from nn import FeedforwardNN

# Sweep Configuration
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"values": [0.001, 0.0001]},
        "activation": {"values": ["sigmoid", "tanh", "relu"]},
        "optimizer": {"values": ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]},
        "epochs": {"values": [5, 10]},
        "num_hidden": {"values": [3, 4, 5]},
        "hidden_size": {"values": [32, 64]},
        "weight_init": {"values": ["random", "xavier"]},
        "batch_size": {"values": [16, 32, 64]},
        "weight_decay": {"values": [0.0, 0.0005, 0.5]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="DA6401 Assignments")

def train_sweep():
    run = wandb.init(project="DA6401 Assignments", entity="da24m010-indian-institute-of-technology-madras") 
    config = wandb.config 
    run.name = f"LR_{config.learning_rate}_HL_{config.num_hidden}_HLS_{config.hidden_size}_OPT_{config.optimizer}_ACTIVATION_{config.activation}_NUM_EPOCHS_{config.epochs}_BATCH_SIZE_{config.batch_size}_W_INIT_{config.weight_init}_W_DECAY_{config.weight_decay}"

    model = FeedforwardNN(num_layers=config.num_hidden, hidden_size=config.hidden_size, learning_rate=config.learning_rate,
                          activation=config.activation, optimizer=config.optimizer, weight_init=config.weight_init, weight_decay=config.weight_decay, loss_function="mse")

    model.train(epochs=config.epochs, batch_size=config.batch_size, dataset = "fashion_mnist", wandb_logs = True)
    
wandb.agent(sweep_id, function=train_sweep)