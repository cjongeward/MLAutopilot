# my_script.py
import torch
import torch.nn as nn
import os
from HdgHoldModel import HdgHoldNN
from HdgAltHoldModel import HdgAltHoldNN
from ILSModel import ILSModelNN
from ILSModel import LocModelNN
from RunwayTracker import RunwayTrackerModelNN
import cv2



# Load the trained model
model = RunwayTrackerModelNN()
#cwd = os.getcwd()
#modelPath = os.path.join(cwd, "..", "models", model.ModelFileName)
model.load_state_dict(torch.load(f"C:/repos/MLAutopilot/models/{model.ModelFileName}"))
#model.load_state_dict(torch.load(modelPath))
model.eval()  # Set to evaluation mode

# Example input for inference
def predict(hdg, dhdg, bank):
    example_input = torch.tensor([hdg, dhdg, bank], dtype=torch.float32).view(-1, 3)
    with torch.no_grad():
        predicted_output = model(example_input)
    return predicted_output[0].item()
def predictHdgAlt(ias, alt, hdg, dalt, dhdg, bank, pitch):
    example_input = torch.tensor([ias, alt, hdg, dhdg, dalt, bank, pitch], dtype=torch.float32).view(-1, 7)
    with torch.no_grad():
        predicted_output = model(example_input)
    return [predicted_output[0, 0].item(), predicted_output[0, 1].item()]

def predictILS(ias, alt, hdg, bank, pitch, loc, gs):
    example_input = torch.tensor([ias, alt, hdg, bank, pitch, loc, gs], dtype=torch.float32).view(-1, 7)
    with torch.no_grad():
        predicted_output = model(example_input)
    return [predicted_output[0, 0].item(), predicted_output[0, 1].item(), predicted_output[0,2].item() ]

def predictLoc(hdg, bank, loc):
    example_input = torch.tensor([hdg, bank, loc], dtype=torch.float32).view(-1, 3)
    with torch.no_grad():
        predicted_output = model(example_input)
    return [predicted_output[0, 0].item()]

def predictRT(hdg, bank, alt):
    example_input = torch.tensor([hdg, bank, alt], dtype=torch.float32).view(-1, 3)

    cap = cv2.VideoCapture(2)  # Adjust device index as needed
    if not cap.isOpened():
        print("Cannot open camera")
        return None

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return None

    # Assuming a preloaded PyTorch model and preprocess function exists
    frame = cv2.resize(frame, (256, 256))  # Example resizing
    frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)  # HWC to CHW
    frame = frame / 255.0  # Normalization

    with torch.no_grad():
        predicted_output = model(frame.unsqueeze(0), example_input)
    return [predicted_output[0, 0].item(), predicted_output[0, 1].item()]
