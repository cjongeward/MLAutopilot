# my_script.py
import torch
import torch.nn as nn

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


# Load the trained model
model = SimpleNN()
model.load_state_dict(torch.load('C:/MSFS SDK/Samples/VisualStudio/SimvarWatcher/simple_nn_model.pth'))
model.eval()  # Set to evaluation mode

# Example input for inference
def predict(bank, hdg):
    example_input = torch.tensor([bank, hdg], dtype=torch.float32).view(-1, 2) / 180.0
    with torch.no_grad():
        predicted_output = model(example_input)
    return predicted_output[0].item()
