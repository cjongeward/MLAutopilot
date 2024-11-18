import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
import os
from PIL import Image
from ImageExtractor import extract_frames

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
class CSVImageDataset(Dataset):
    def __init__(self, csv_file, image_dir, input_names, output_names):
        # Load the CSV data
        self.data = pd.read_csv(csv_file)
        self.input_indices = [DataIndices[n] for n in input_names]
        self.output_indices = [DataIndices[n] for n in output_names]
        self.image_dir = image_dir
        video_path = os.path.join(self.image_dir, "..", "video.mkv")
        timestamps = self.data.iloc[:, -1].unique().tolist()
        extract_frames(video_path, timestamps, image_dir)

    def __len__(self):
        # Return the number of data samples
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Separate the input and output
        X = torch.tensor(self.data.iloc[idx, self.input_indices], dtype=torch.float32)
        y = torch.tensor(self.data.iloc[idx, self.output_indices], dtype=torch.float32)
        timestamp = self.data.iloc[idx, -1]
        img_name = os.path.join(self.image_dir, f"{timestamp}ms.png")
        image = Image.open(img_name).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)
        return X, image, y


def getDataset(filename, input_names, output_names):
    cwd = os.getcwd()
    path = os.path.join(cwd, "..", "data", filename)
    image_dir = os.path.join(cwd, "..", "data", "temp")
    return CSVImageDataset(path, image_dir, input_names, output_names)

def splitDataset(dataset, test_size):
    train_size = int(len(dataset) - test_size)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

class RunwayTrackerModelNN(nn.Module):
    InputNames = ["hdg", "bank", "alt"]
    OutputNames = ["aileronPos", "elevatorPos"]
    DatasetFilename = "testdata.csv"
    ModelFileName = "testModel.pth"

    def __init__(self):
        super(RunwayTrackerModelNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1_img = nn.Linear(32 * 64 * 64, 512)

        self.l1 = nn.Linear(len(RunwayTrackerModelNN.InputNames), 32)  # Input to hidden layer

        self.combined_l2 = nn.Linear(32 + 512, 128)  # Hidden layer to output
        self.final_l = nn.Linear(128 ,len(RunwayTrackerModelNN.OutputNames))  # Hidden layer to output
        #nn.init.xavier_uniform_(layer1.weight)
        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None

    def load_datasets(self):
        self.dataset = getDataset(RunwayTrackerModelNN.DatasetFilename, RunwayTrackerModelNN.InputNames, RunwayTrackerModelNN.OutputNames)
        self.train_dataset, self.test_dataset = splitDataset(dataset=self.dataset, test_size=40)

    def forward(self, img, x):
        x_img = self.pool(F.relu(self.conv1(img)))
        x_img = self.pool(F.relu(self.conv2(x_img)))
        x_img = x_img.view(-1, 32 * 64 * 64)
        x_img = F.relu(self.fc1_img(x_img))

        x_data = F.relu(self.l1(x))

        x = torch.cat((x_img, x_data), dim=1)
        x = F.relu(self.combined_l2(x))
        yhat = self.final_l(x)
        return yhat


