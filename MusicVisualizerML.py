import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import joblib

# Load the example data with RGB values
data = pd.read_csv('train_1.csv')

# Separate features and labels
data['currFreq_weighted'] = data['currFreq']
X = data[['currFreq_weighted', 'freq1', 'freq2', 'freq3', 'freq4', 'freq5',
          'freq6', 'freq7', 'freq8', 'freq9', 'freq10']].values
y = data[['r', 'g', 'b']].values

# Normalize the input data
scaler = StandardScaler()
X = scaler.fit_transform(X)

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


# Model parameters
input_size = X.shape[1]
hidden_size = 16  # Reduced from 32 to 16
output_size = 3  # For RGB values

model = FrequencyToColorModel(input_size, hidden_size, output_size)

y = y / 255.0

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Redefine the dataset and data loader
dataset = TensorDataset(X_tensor, y_tensor)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Define the model, loss function, and optimizer as before
model = FrequencyToColorModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with increased epochs
num_epochs = 100  # Increased from 20 to 100
for epoch in range(num_epochs):
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save model parameters
torch.save(model.state_dict(), 'model_parameters_rgb.pth')

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# Function to extract parameters


def extract_parameters(model):
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.detach().numpy()
    return params


params = extract_parameters(model)

# Print specific parameters for fc1 and fc2
print("\nSpecific parameters:")
print(f"fc1.weight:\n{params['fc1.weight']}\n")
print(f"fc1.bias:\n{params['fc1.bias']}\n")
print(f"fc2.weight:\n{params['fc2.weight']}\n")
print(f"fc2.bias:\n{params['fc2.bias']}\n")

print("Scale:", scaler.scale_)
print("mean:", scaler.mean_)


# Inference with the same data
model.eval()
with torch.no_grad():
    # Scale outputs back to original RGB range
    outputs = model(X_tensor) * 255.0
    predicted_rgb = outputs.numpy()

# Compare predictions with actual values
comparison = pd.DataFrame({
    'Actual R': y[:, 0] * 255.0,
    'Predicted R': predicted_rgb[:, 0],
    'Actual G': y[:, 1] * 255.0,
    'Predicted G': predicted_rgb[:, 1],
    'Actual B': y[:, 2] * 255.0,
    'Predicted B': predicted_rgb[:, 2]
})

print(comparison.head())
