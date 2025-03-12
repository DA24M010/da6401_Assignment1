import wandb
from nn import FeedforwardNN

run = wandb.init(project="DA6401 Assignments", entity="da24m010-indian-institute-of-technology-madras", name = 'best_model_test_accuracy_and_confusion_matrix') 
model = FeedforwardNN(num_layers=3, hidden_size=64, learning_rate=0.001,
                        activation='tanh', optimizer='nadam', weight_init='xavier', weight_decay=0)

model.train(epochs=10, batch_size=64, dataset = "fashion_mnist", wandb_logs= True, log_test= True)