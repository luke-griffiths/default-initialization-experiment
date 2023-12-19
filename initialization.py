import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Iterable, Any, Dict

device = torch.device("mps") if torch.backends.mps.is_available() and torch.backends.mps.is_built() else torch.device("cpu")
NUM_EPOCHS = 15
BATCH_SIZE = 128
LR = 0.0001

class ReLUNetwork(nn.Module):
    """
    Simple model with ReLU activations for MNIST
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=784, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10)
        )
    def forward(self, x):
        return self.model(x)

### HELPER METHODS ### 
def load_MNIST_dataset():
	train_dataset = torchvision.datasets.MNIST(
		root = './data',
		train = True,
		transform = torchvision.transforms.ToTensor(),
		download = True)
	test_dataset = torchvision.datasets.MNIST(
		root = './data',
		train = False,
		transform = torchvision.transforms.ToTensor(),
		download = False)
	return (train_dataset, test_dataset)


def construct_dataloaders(train_dataset, test_dataset, batch_size, shuffle_train=True):
	train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
	test_dl = DataLoader(test_dataset, batch_size=50, shuffle=False)

	return train_dl, test_dl


@torch.no_grad()
def evaluate_model(dataloader, model, loss_fn):
    """
    returns (avg loss, accuracy) of model
    """
    total_acc, total_loss = 0.0, 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).to(device)
        loss = loss_fn(preds, labels)
        total_loss += loss.item() 
        preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)
        total_acc += torch.sum(preds == labels).item()

    return total_loss / len(dataloader.dataset), total_acc / len(dataloader.dataset)


def train(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs, eval_train_stats=True, eval_test_stats=True):
    """
    trains a model and returns lists of train approx loss/acc, train/test actual loss/acc
    """
    train_loss, train_acc = [], []
    test_loss, test_acc = [], []
    approx_tr_loss, approx_tr_acc = [], []

    for _ in range(epochs):

        model.train()
        epoch_loss, epoch_acc = 0.0, 0.0
        for images, labels in tqdm(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images).to(device)
            loss = loss_fn(preds, labels)
            epoch_loss += loss.item()
            preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)
            epoch_acc += torch.sum(preds == labels)
            loss.backward()
            optimizer.step()

        # contains averages of all examples in this epoch
        approx_tr_loss.append(epoch_loss / len(train_dataloader))
        approx_tr_acc.append(epoch_acc.to("cpu") / len(train_dataloader))

        model.eval()
        # contains total
        if eval_train_stats:
            tloss, tacc = evaluate_model(train_dataloader, model, loss_fn)
            train_loss.append(tloss)
            train_acc.append(tacc)

        if eval_test_stats:
            tloss, tacc = evaluate_model(test_dataloader, model, loss_fn)
            test_loss.append(tloss)
            test_acc.append(tacc)

    return train_loss, train_acc, test_loss, test_acc, approx_tr_loss, approx_tr_acc


def make_plot(title : str, xlabel : str, ylabel : str, x : Iterable[Any], y_dict : Dict[str, Iterable[Any]]):
    """
    Creates and saves a multi-curve plot

    param title = title displayed on the plot
    param xlabel = label displayed on the x-axis
    param ylabel = label displayed on the y-axis
    param x = x coordinates of input data
    param y_dict = dictionary mapping curve label -> curve y data
    """
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for name, ys in y_dict.items():
        plt.plot(x, ys, label=name)
    plt.legend()
    plt.savefig(f"{title}.png")


def print_model_statistics(model : nn.Module):
    """
    Convenience method just to print out any relevant architecture data
    """
    print(f"Model Architecture\n{model}\n")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of Parameters: {num_params}\n")

    print(f"Layer-wise Information:")
    for name, param in model.named_parameters():
        print(f"Layer: {name}, Size: {param.size()}, Number of Parameters: {param.numel()}")
    print("\n")


def make_model_weights_histogram(title : str, label : str, model : nn.Module):
    """
    Sanity check method to plot a histogram of the model's weights
    param title = title of the plot
    param label = label of the data shown in the plot's legend
    param model = nn model
    """
    learnable_weights = [param.data.cpu().numpy() for param in model.parameters() if param.requires_grad]

    flat_learnable_weights = np.concatenate([weight.flatten() for weight in learnable_weights])

    plt.figure()
    plt.title(f"{title} Weight Histogram")
    plt.xlabel("Weight")
    plt.hist(flat_learnable_weights, bins=100, density=True, alpha=0.7, label=label)
    plt.legend()
    plt.savefig(f"{title}.png")


def initialize_with_strategy(func):
    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            func(m.weight)
            
    return initialize_weights


def run_experiment(train_dl, test_dl, initializations):

    def initialize_and_train_model(initialization_name, init_func):

        model = ReLUNetwork().to(device)
        if init_func is not None:
            model.apply(initialize_with_strategy(init_func))

        # plot the weights before training, as a sanity check
        make_model_weights_histogram(title=initialization_name, label=initialization_name, model=model)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        train_loss, train_acc, test_loss, test_acc, _, _ = train(train_dl, test_dl, model, nn.CrossEntropyLoss(), optimizer, NUM_EPOCHS)

        return train_loss, test_loss, train_acc, test_acc
    
    results = {
        "train_loss" : {},
        "test_loss" : {},
        "train_acc" : {},
        "test_acc" : {}
    }
    
    # run with all initializations
    for name, func in initializations.items():
        print(f"Initializing model with {name} initialization and training for {NUM_EPOCHS} epochs")

        train_loss, test_loss, train_acc, test_acc = initialize_and_train_model(name, func)

        results["train_loss"][name] = train_loss
        results["test_loss"][name] = test_loss
        results["train_acc"][name] = train_acc
        results["test_acc"][name] = test_acc
        
    # generate all plots
    epochs = [i for i in range(NUM_EPOCHS)]
    make_plot("Train Loss", "Epoch", "Loss", epochs, results["train_loss"])
    make_plot("Test Loss", "Epoch", "Loss", epochs, results["test_loss"])
    make_plot("Train Accuracy", "Epoch", "Accuracy", epochs, results["train_acc"])
    make_plot("Test Accuracy", "Epoch", "Accuracy", epochs, results["test_acc"])

    print(f"Experiment finished.")


if __name__ == "__main__":
    torch.manual_seed(0)
    # create separate MNIST train/test sets
    (train_dataset, test_dataset) = load_MNIST_dataset()
    # create their dataloaders
    train_dl, test_dl = construct_dataloaders(train_dataset, test_dataset, batch_size=BATCH_SIZE)

    # define the initialization strategies we want to test
    initializations = {
        "PyTorch Default" : None,
        "Xavier Uniform" : nn.init.xavier_uniform_,
        "Xavier Normal" : nn.init.xavier_normal_,
        "He Uniform" : nn.init.kaiming_uniform_,
        "He Normal" : nn.init.kaiming_normal_
    }
    
    run_experiment(train_dl, test_dl, initializations)