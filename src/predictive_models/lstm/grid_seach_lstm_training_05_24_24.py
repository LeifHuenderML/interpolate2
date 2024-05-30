import time
import lstm
import json
import torch
import queue
import threading
import numpy as np 
import pandas as pd
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

##running this script will result on trining a model on 864 varation looking for the best set of hyperparameters

dataset = torch.load('../../data/pecan/lstm_weather_dataset.pt')


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
new_val_size = int(0.8 * val_size)
test_size = val_size - new_val_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
val_dataset, test_dataset = random_split(val_dataset, [new_val_size, test_size], generator=generator)

print(f'Train Dataset size: {len(train_dataset)} \nValidation Dataset size {len(val_dataset)}\nTest Dataset size {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

hidden_size = [16, 32, 64, 128, 256, 512]
num_layers = [1, 2, 3, 4, 5, 6,]
criterion = [nn.MSELoss(), nn.L1Loss(), nn.HuberLoss(), nn.SmoothL1Loss()]
lr = [1, 0.3, 0.1, 0.03, 0.01, 0.001,]


hyperparameters = {
    'hidden_size' : hidden_size,
    'num_layers' : num_layers,
    'criterion' : criterion,
    'lr' : lr
}

# set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device set to:", device)

grid_search = lstm.GridSearch(hyperparameters, 3, train_loader, val_loader, test_loader, device)
start= time.time()
grid_search.search(hyperparameters)



with open('lstm_training_time.txt', 'w') as f:
    f.write(str(time.time() - start))
