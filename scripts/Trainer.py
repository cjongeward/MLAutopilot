import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

import matplotlib.pyplot as plt
import os

# Define a Dataset class
class CSVDataset(Dataset):
    def __init__(self, csv_file):
        # Load the CSV data
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        # Return the number of data samples
        return len(self.data)

    def __getitem__(self, idx):
        # Separate the input and output
        X = torch.tensor(self.data.iloc[idx, [0,1]], dtype=torch.float32)
        y = torch.tensor(self.data.iloc[idx, 2], dtype=torch.float32)
        return X, y

# Load the dataset and create a DataLoader
cwd = os.getcwd()
dataset = CSVDataset(os.path.join(cwd, "..", "data", "data.csv"))
train_ratio = 0.9
train_size = int(len(dataset) * train_ratio)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"device: {device}")

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        layer1 = nn.Linear(2, 64)  # Input to hidden layer
        layer2 = nn.Linear(64, 64)  # Hidden layer to output
        layer3 = nn.Linear(64, 64)  # Hidden layer to output
        layer4 = nn.Linear(64 ,1)  # Hidden layer to output
        nn.init.xavier_uniform_(layer1.weight)
        nn.init.xavier_uniform_(layer2.weight)
        nn.init.xavier_uniform_(layer3.weight)
        nn.init.xavier_uniform_(layer4.weight)
        self.lin_relu_stack = nn.Sequential(
            layer1,
            nn.ReLU(),
            layer2,
            nn.ReLU(),
            layer3,
            nn.ReLU(),
            layer4
        )

    def forward(self, x):
        return self.lin_relu_stack(x)
        #x = torch.relu(self.layer1(x))
        #x = self.layer2(x)
        #return x


model = SimpleNN()
criterion = nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.000001)
optimizer = torch.optim.Adam(model.parameters())

num_epochs = 200

model.train()
for epoch in range(num_epochs):
    for inputs, targets in train_dataloader:
        # Reshape inputs to match model expectations
        inputs = inputs.view(-1, 2)# / 180.0
        inputs[:,0] = inputs[:,0] / 180.0
        inputs[:,1] = inputs[:,1] / 180.0
        targets = targets.view(-1, 1)

        # Forward pass
        #outputs = model.forward(inputs)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        #print(f"model.params: {model.lin_relu_stack[0].weight}")

model.eval()
with torch.no_grad():
    avg_loss = 0.0
    for inputs, targets in train_dataloader:
        inputs = inputs.view(-1, 2)
        inputs[:,0] = inputs[:,0] / 180.0
        inputs[:,1] = inputs[:,1] / 180.0
        targets = targets.view(-1, 1)
        pred = model(inputs)
        losses = abs(targets - pred)
        avg_loss = sum(losses)
        #for p, t, l in zip(pred, targets, losses):
            #print(f"pred: {p}, target: {t}, loss: {l}")
    avg_loss = avg_loss / len(test_dataset)
    print(f"Average train loss: {avg_loss}")


model.eval()
plotme = True
with torch.no_grad():
    avg_loss = 0.0
    for inputs, targets in test_dataloader:
        inputs = inputs.view(-1, 2)
        inputs[:,0] = inputs[:,0] / 180.0
        inputs[:,1] = inputs[:,1] / 180.0
        targets = targets.view(-1, 1)
        pred = model(inputs)
        losses = abs(targets - pred)
        avg_loss = sum(losses)
        print("test loop")
        #for i, (p, t, l) in enumerate(zip(pred, targets, losses)):
        #    if i % 10 == 0:
        #        print(f"pred: {p}, target: {t}, loss: {l}")

        if plotme == True:
            xp = inputs.numpy()
            yp = targets.numpy()
            pp = pred.numpy()
            plt.scatter(xp[:,0], yp, label="Actual bank vs ail")
            plt.scatter(xp[:,0], pp, color="red", label="Predictions")
            plt.xlabel("bank")
            plt.ylabel("ail")
            plt.legend()
            plt.show()

    avg_loss = avg_loss / len(test_dataset)
    print(f"Average test loss: {avg_loss}")

with torch.no_grad():
    avg_loss = 0.0
    test_inputs = torch.tensor([[0.0, 90.], [0.0, 0.0], [0.0, -90.0], [30.0, 90.0], [-30.0, 90.0]])
    for inputs in test_inputs:
        inputs = inputs.view(-1, 2)
        inputs[:,0] = inputs[:,0] / 180.0
        inputs[:,1] = inputs[:,1] / 180.0
        pred = model(inputs)
        print(f"pred: {pred[0]}, bank: {inputs[0,0]}, hdg: {inputs[0,1]}")



torch.save(model.state_dict(), 'simple_nn_model.pth')


