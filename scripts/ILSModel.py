import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

import matplotlib.pyplot as plt
import os

DataIndices = {
    "ias": 0,
    "alt": 1,
    "hdg": 2,
    "dias": 3,
    "dalt": 4,
    "dhdg": 5,
    "bank": 6,
    "pitch": 7,
    "aileronPos": 8,
    "elevatorPos": 9,
    "rudderPos": 10,
    "throttlePos": 11,
    "loc": 12,
    "gs": 13,
    "elevatorTrimPos": 14
}
class CSVDataset(Dataset):
    def __init__(self, csv_file, input_names, output_names):
        # Load the CSV data
        self.data = pd.read_csv(csv_file)
        self.input_indices = [DataIndices[n] for n in input_names]
        self.output_indices = [DataIndices[n] for n in output_names]

    def __len__(self):
        # Return the number of data samples
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Separate the input and output
        X = torch.tensor(self.data.iloc[idx, self.input_indices], dtype=torch.float32)
        y = torch.tensor(self.data.iloc[idx, self.output_indices], dtype=torch.float32)
        return X, y


def getDataset(filename, input_names, output_names):
    cwd = os.getcwd()
    path = os.path.join(cwd, "..", "data", filename)
    return CSVDataset(path, input_names, output_names)

def splitDataset(dataset, test_size):
    train_size = int(len(dataset) - test_size)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

class LocModelNN(nn.Module):
    InputNames = ["hdg", "bank", "loc"]
    OutputNames = ["aileronPos"]
    DatasetFilename = "LocData.csv"
    ModelFileName = "LocModel.pth"

    def __init__(self):
        super(LocModelNN, self).__init__()
        layer1 = nn.Linear(len(LocModelNN.InputNames), 32)  # Input to hidden layer
        layer2 = nn.Linear(32, 32)  # Hidden layer to output
        layer3 = nn.Linear(32, 32)  # Hidden layer to output
        layer4 = nn.Linear(32, 32)  # Hidden layer to output
        layer5 = nn.Linear(32 ,len(LocModelNN.OutputNames))  # Hidden layer to output
        nn.init.xavier_uniform_(layer1.weight)
        nn.init.xavier_uniform_(layer2.weight)
        nn.init.xavier_uniform_(layer3.weight)
        nn.init.xavier_uniform_(layer4.weight)
        nn.init.xavier_uniform_(layer5.weight)
        self.lin_relu_stack = nn.Sequential(
            layer1,
            nn.ReLU(),
            layer2,
            nn.ReLU(),
            layer3,
            nn.ReLU(),
            layer4,
            nn.ReLU(),
            layer5
        )
        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None

    def load_datasets(self):
        self.dataset = getDataset(LocModelNN.DatasetFilename, LocModelNN.InputNames, LocModelNN.OutputNames)
        self.train_dataset, self.test_dataset = splitDataset(dataset=self.dataset, test_size=40)

    def forward(self, x):
        return self.lin_relu_stack(x)




class ILSModelNN(nn.Module):
    InputNames = ["ias", "alt", "hdg", "bank", "pitch", "loc", "gs"]
    OutputNames = ["aileronPos", "elevatorPos", "throttlePos"]
    DatasetFilename = "ILSData.csv"
    ModelFileName = "ILSModel.pth"

    def __init__(self):
        super(ILSModelNN, self).__init__()
        layer1 = nn.Linear(len(ILSModelNN.InputNames), 64)  # Input to hidden layer
        layer2 = nn.Linear(64, 64)  # Hidden layer to output
        layer3 = nn.Linear(64, 64)  # Hidden layer to output
        layer4 = nn.Linear(64, 64)  # Hidden layer to output
        layer5 = nn.Linear(64 ,len(ILSModelNN.OutputNames))  # Hidden layer to output
        nn.init.xavier_uniform_(layer1.weight)
        nn.init.xavier_uniform_(layer2.weight)
        nn.init.xavier_uniform_(layer3.weight)
        nn.init.xavier_uniform_(layer4.weight)
        nn.init.xavier_uniform_(layer5.weight)
        self.lin_relu_stack = nn.Sequential(
            layer1,
            nn.ReLU(),
            layer2,
            nn.ReLU(),
            layer3,
            nn.ReLU(),
            layer4,
            nn.ReLU(),
            layer5
        )
        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None

    def load_datasets(self):
        self.dataset = getDataset(ILSModelNN.DatasetFilename, ILSModelNN.InputNames, ILSModelNN.OutputNames)
        self.train_dataset, self.test_dataset = splitDataset(dataset=self.dataset, test_size=40)

    def forward(self, x):
        return self.lin_relu_stack(x)


