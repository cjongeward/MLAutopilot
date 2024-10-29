# my_script.py
import torch
import torch.nn as nn
import os
from HdgHoldModel import HdgHoldNN

# Load the trained model
model = HdgHoldNN()
#cwd = os.getcwd()
#modelPath = os.path.join(cwd, "..", "models", model.ModelFileName)
model.load_state_dict(torch.load('C:/repos/MLAutopilot/models/hdgHoldModel.pth'))
#model.load_state_dict(torch.load(modelPath))
model.eval()  # Set to evaluation mode

# Example input for inference
def predict(hdg, dhdg, bank):
    example_input = torch.tensor([hdg, dhdg, bank], dtype=torch.float32).view(-1, 3)
    with torch.no_grad():
        predicted_output = model(example_input)
    return predicted_output[0].item()
