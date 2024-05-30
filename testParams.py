import math
import torch
import torch.nn as nn
import pandas as pd
import joblib

# Define the model


class FrequencyToColorModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FrequencyToColorModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # Round to the nearest integer
        return x


def test_model(test_csv_path, model, scaler):
    # Load the test data
    test_data = pd.read_csv(test_csv_path)

    # Preprocess the test data
    test_data['currFreq_weighted'] = test_data['currFreq']
    X_test = test_data[['currFreq_weighted', 'freq1', 'freq2', 'freq3', 'freq4', 'freq5',
                        'freq6', 'freq7', 'freq8', 'freq9', 'freq10']].values
    X_test = scaler.transform(X_test)  # Normalize the test data

    # Convert the test data to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Get predictions
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Scale outputs back to original RGB range
        outputs = model(X_test_tensor) * 255.0
        predicted_rgb = outputs.numpy()

    # Print predictions
    for i, prediction in enumerate(predicted_rgb):
        print(
            f'Prediction {i+1}: R={int(prediction[0])}, G={int(prediction[1])}, B={int(prediction[2])}')


# Load the scaler and model parameters
scaler = joblib.load('scaler.pkl')
model = FrequencyToColorModel(input_size=11, hidden_size=16, output_size=3)
model.load_state_dict(torch.load('model_parameters_rgb.pth'))

# Test the model with the new test data
test_model('test_1.csv', model, scaler)
