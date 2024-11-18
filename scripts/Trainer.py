import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

import matplotlib.pyplot as plt
import os

from HdgHoldModel import HdgHoldNN
from HdgAltHoldModel import HdgAltHoldNN
from ILSModel import ILSModelNN
from ILSModel import LocModelNN
from RunwayTracker import RunwayTrackerModelNN


# Define a Dataset class



def saveModel(model):
    cwd = os.getcwd()
    modelPath = os.path.join(cwd, "..", "models", model.ModelFileName)
    torch.save(model.state_dict(), modelPath)


def trainModel(model, dataset, num_epochs=100, criterion = nn.MSELoss(), print_status=True):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs = inputs.view(-1, len(model.InputNames))
            targets = targets.view(-1, len(model.OutputNames))
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print loss every 10 epochs
        if print_status and (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.1f}')

def evaluateModel(model, dataset, datasetName):
    model.eval()
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    with torch.no_grad():
        avg_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.view(-1, len(model.InputNames))
            targets = targets.view(-1, len(model.OutputNames))
            pred = model(inputs)
            losses = abs(targets - pred)
            avg_loss = sum(losses)
        avg_loss = avg_loss / len(dataset)
        print(f"Average loss {datasetName}: {avg_loss}")


def trainModelImage(model, dataset, num_epochs=100, criterion = nn.MSELoss(), print_status=True):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    for epoch in range(num_epochs):
        for inputs, image, targets in dataloader:
            inputs = inputs.view(-1, len(model.InputNames))
            targets = targets.view(-1, len(model.OutputNames))
            outputs = model(image, inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print loss every 10 epochs
        if print_status and (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.1f}')

def evaluateModelImage(model, dataset, datasetName):
    model.eval()
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    with torch.no_grad():
        avg_loss = 0.0
        for inputs, image, targets in dataloader:
            inputs = inputs.view(-1, len(model.InputNames))
            targets = targets.view(-1, len(model.OutputNames))
            pred = model(image, inputs)
            losses = abs(targets - pred)
            avg_loss = sum(losses)
        avg_loss = avg_loss / len(dataset)
        print(f"Average loss {datasetName}: {avg_loss}")

def spotCheck(model):
    model.eval()
    with torch.no_grad():
        #example_input = torch.tensor([ias, alt, hdg, bank, pitch, loc, gs], dtype=torch.float32).view(-1, 7)
        example_input = torch.tensor([100.0, 1600.0, 220.0, -20.0, 0.0, -127, -119], dtype=torch.float32).view(-1, 7)
        pred = model(example_input)
        print(f"right of course, right of hdg, right bank: elev, ail, throt {pred}")
        example_input = torch.tensor([100.0, 1600.0, 220.0, 30.0, 0.0, -127, -119], dtype=torch.float32).view(-1, 7)
        pred = model(example_input)
        print(f"right of course, right of hdg, left bank: elev, ail, throt {pred}")
        example_input = torch.tensor([100.0, 1600.0, 180.0, 0.0, 0.0, -127, -119], dtype=torch.float32).view(-1, 7)
        pred = model(example_input)
        print(f"right of course, intercept, level: elev, ail, throt {pred}")
        example_input = torch.tensor([100.0, 1600.0, 180.0, 0.0, 0.0, -10, -119], dtype=torch.float32).view(-1, 7)
        pred = model(example_input)
        print(f"right and correcting, intercept, level: elev, ail, throt {pred}")


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"device: {device}")

model = RunwayTrackerModelNN()
model.load_datasets()
trainModelImage(model=model, dataset=model.train_dataset)
evaluateModelImage(model=model, dataset=model.train_dataset, datasetName="Training")
evaluateModelImage(model=model, dataset=model.test_dataset, datasetName="Test")
#spotCheck(model)

saveModel(model)




#        if plotme == True:
#            xp = inputs.numpy()
#            yp = targets.numpy()
#            pp = pred.numpy()
#            plt.scatter(xp[:,0], yp, label="Actual bank vs ail")
#            plt.scatter(xp[:,0], pp, color="red", label="Predictions")
#            plt.xlabel("bank")
#            plt.ylabel("ail")
#            plt.legend()
#            plt.show()




