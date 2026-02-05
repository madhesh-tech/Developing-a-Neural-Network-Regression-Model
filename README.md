# Developing a Neural Network Regression Model
## Name: Madhesh I
## Register Number: 212224220055

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.
<img width="842" height="592" alt="image" src="https://github.com/user-attachments/assets/4b9505a5-0e32-4224-933d-e3c7691bc7af" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM
```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('/content/Book1.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Name: MADHESH I
# Register Number: 212224220055
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}

  def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x
# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.Adam(ai_brain.parameters(),lr=0.001)

# Name: MADHESH I
# Register Number: 212224220055
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    # Write your code here
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(ai_brain.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

```
### Dataset Information
<img width="172" height="290" alt="image" src="https://github.com/user-attachments/assets/dc3aa4c1-754b-4008-a552-134d45e4db5a" />

### OUTPUT
<img width="383" height="56" alt="Screenshot 2026-02-05 135744" src="https://github.com/user-attachments/assets/b6c08790-fdfd-47cd-9793-2139ee19531e" />
<img width="435" height="250" alt="image" src="https://github.com/user-attachments/assets/43d3fcbe-1b1a-45d7-85bf-6feeaa57e430" />

### Training Loss Vs Iteration Plot
<img width="828" height="577" alt="image" src="https://github.com/user-attachments/assets/b592f63e-67c1-43cc-9f28-29761b99c0cc" />

### New Sample Data Prediction
<img width="437" height="62" alt="Screenshot 2026-02-05 135759" src="https://github.com/user-attachments/assets/d14d971a-c5f4-413d-ac13-40d713783673" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
