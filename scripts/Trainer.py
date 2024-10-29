import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

import matplotlib.pyplot as plt
import os

from HdgHoldModel import HdgHoldNN
from HdgAltHoldModel import HdgAltHoldNN


# Define a Dataset class



def saveModel(model):
    cwd = os.getcwd()
    modelPath = os.path.join(cwd, "..", "models", model.ModelFileName)
    torch.save(model.state_dict(), modelPath)


def trainModel(model, dataset, num_epochs=200, criterion = nn.MSELoss(), print_status=True):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
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


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"device: {device}")

model = HdgAltHoldNN()
model.load_datasets()
trainModel(model=model, dataset=model.train_dataset)
evaluateModel(model=model, dataset=model.train_dataset, datasetName="Training")
evaluateModel(model=model, dataset=model.test_dataset, datasetName="Test")
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




