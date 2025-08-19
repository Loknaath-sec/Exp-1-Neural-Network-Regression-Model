# EXP-1: Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

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

## PROGRAM
### Name: LOKNAATH P
### Register Number: 212223240080
```python
import pandas as pd
data=pd.read_csv("/content/height.csv")
data

X=data[['height']]
Y=data[['weight']]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=33)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

import torch

X_train_tensor=torch.tensor(X_train,dtype=torch.float32)
Y_train_tensor=torch.tensor(Y_train.values,dtype=torch.float32).view(-1,1)
X_test_tensor=torch.tensor(X_test,dtype=torch.float32)
Y_test_tensor=torch.tensor(Y_test.values,dtype=torch.float32).view(-1,1)

import torch.nn as nn
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

Loknaath_brain = NeuralNet()
criterion=nn.MSELoss()
optimizer=torch.optim.RMSprop(Loknaath_brain.parameters(),lr=0.001)

def train_model(Loknaath_brain,X_train,y_train,criterion,optimizer,epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion(Loknaath_brain(X_train),y_train)
    loss.backward()
    optimizer.step()

    Loknaath_brain.history['loss'].append(loss.item())
    if epoch % 200 == 0:
      print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

train_model(Loknaath_brain,X_train_tensor,Y_train_tensor,criterion,optimizer)

with torch.no_grad():
  test_loss=criterion(Loknaath_brain(X_test_tensor),Y_test_tensor)
  print(f"Test loss: {test_loss.item():.6f}")

import matplotlib.pyplot as plt
plt.plot(Loknaath_brain.history['loss'])
plt.title(" ")
plt.xlabel("Epochs")
plt.ylabel("Loss")

X_n1_1 = torch.tensor([[50]], dtype=torch.float32)
prediction = Loknaath_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```
## Dataset Information
<img width="907" height="722" alt="image" src="https://github.com/user-attachments/assets/e9523e34-0dd6-4a17-979f-3ef7ab7c017f" />


## OUTPUT

### Training Loss Vs Iteration Plot

<img width="793" height="605" alt="image" src="https://github.com/user-attachments/assets/2d8a3b54-fd18-4de1-a2f0-0c2a8454e7a0" />


### New Sample Data Prediction

<img width="977" height="129" alt="image" src="https://github.com/user-attachments/assets/9849d3fa-a7a7-46d7-bbcf-7a3c17d54f54" />


## RESULT

The program to develop a neural network regression model for the given dataset has been successfully executed.
