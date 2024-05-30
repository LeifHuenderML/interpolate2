# load libraries
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




class LSTM(nn.Module):
    def __init__(self, input_size=17, hidden_size=128, num_layers=2, patience=10, min_delta=10):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
        self.activations = []
        #member vars for the early stopper
        self.patience = patience
        self.min_delta = min_delta
        self.min_validation_loss = float('inf')

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        self.activations.append(out.squeeze())
        return out.squeeze()

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss - self.min_delta):
            self.counter +=1
            if self.counter >= self.patience:
                return True
        return False

class Trainer():
    def __init__(self, model, train_loader, validation_loader, device, criterion=nn.MSELoss(), lr=0.001, num_epochs=1000):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.criterion = criterion
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.train_losses = []
        self.val_losses = []
        self.loss = None

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0
        for data, targets in self.train_loader:
            data = data.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_loader)

    def val_one_epoch(self):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for data, targets in self.validation_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                epoch_loss += loss.item()
        return epoch_loss / len(self.validation_loader)
        
    def train(self):
        for epoch in np.arange(self.num_epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.val_one_epoch()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if(epoch % 100 == 0):
                print(f'Epoch: {epoch} Train Loss: {train_loss} Validation Loss: {val_loss}')

            if self.model.early_stop(val_loss):
                break
            
        print('#'*100)
        print(f'Final Epoch: {epoch} Train Loss: {train_loss} Validation Loss: {val_loss}')
        return self.model, self.train_losses, self.val_losses

class GridSearch():
    def __init__(self, hyperparameters, n_threads, train_loader, validation_loader, test_loader, device, model_weights_path='best_model_weights.pth'):
        self.hyperparameters = hyperparameters
        self.num_threads = n_threads
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.device = device
        self.hidden_size = hyperparameters['hidden_size']
        self.num_layers = hyperparameters['num_layers']
        self.criterion = hyperparameters['criterion']
        self.lr = hyperparameters['lr']
        self.best_model = LSTM()
        self.best_train_losses = []
        self.best_val_losses = []
        self.best_hyperparameters = {}
        self.model_weights_path = model_weights_path
        
    def search(self, hyperparameters):
        iterations = 0
        best_mape = float('inf')
        #super awful loop that brute forces to try and find the best set of hyperparameters
        for hs in hyperparameters['hidden_size']:
            for nl in hyperparameters['num_layers']:
                for crit in hyperparameters['criterion']:
                    for lr in hyperparameters['lr']:

                        #create model and trainer with all selected hyperparameters
                        model = LSTM(hidden_size=hs, num_layers=nl).to(self.device)
                        trainer = Trainer(model, self.train_loader, self.validation_loader, self.device, criterion=crit, lr=lr)
                        #train the model
                        model, train_losses, val_losses = trainer.train()
                        #test to see how good the model performs against a common metric
                        #i had to create a common criteron because since each critrion would provide lightly different accuracy
                        mape = self.test_model(model)
                        best_mape = self.update_best(model, best_mape, mape, hs, nl, crit, lr, train_losses, val_losses)

                        iterations += 1
                        self.display_iterations(iterations)

    
    def test_model(self, model):
        model.eval()
        total_loss = 0
        epsilon = 1e-10

        with torch.no_grad():
            for data, targets in self.test_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                outputs = model(data)
                loss = torch.abs((targets - outputs) / (targets + epsilon)).mean()
                total_loss += loss.item() * targets.size(0) 

        average_loss = total_loss / len(self.test_loader.dataset)
        mape = average_loss * 100 
        return mape
    
    def update_best(self, model, best_mape, mape, hs, nl, crit, lr, train_losses, val_losses):
        if best_mape > mape:
            print(f'Best Accuracy: {mape}')
            best_mape = mape
            self.best_model = model
            torch.save(model.state_dict(), self.model_weights_path)
            self.best_train_losses = train_losses
            self.best_val_losses = val_losses
            self.best_hyperparams = {
                'hidden_size' : hs,
                'num_layers' : nl,
                'criterion' : crit,
                'lr' : lr
            }
            result = self.make_results()
            self.save_results(result)

        return best_mape
    
    def make_results(self,):
        return {
            'model': self.best_model,
            'train_losses': self.best_train_losses,
            'val_losses' : self.best_val_losses,
            'hyperparameters' : self.best_hyperparams
        }
    
    def save_results(self, results):
        hyperparameters = results['hyperparameters']
        loss_fn_name = hyperparameters['criterion'].__class__.__name__
        hyperparameters['criterion'] = loss_fn_name
        with open('lstm_hyperparameters.json', 'w') as f:
            json.dump(hyperparameters, f)
            
    def display_iterations(self, iterations):
        if iterations % 100 == 0:
            print("#"*100)
            print()
            print(f'Iteration: {iterations}')
            print()
            print("#"*100)

    def multi_thread(self):
        threads = []
        results = queue.Queue()
        hyperparameter_lengths = {
            'hidden_size': len(self.hidden_size),
            'num_layers': len(self.num_layers),
            'criterion': len(self.criterion),
            'lr': len(self.lr)
        }

        for i in range(self.num_threads):
            begin = i
            end = i + 1
            hyperparameters = {}
            
            for key, length in hyperparameter_lengths.items():
                if begin < length:
                    hyperparameters[key] = getattr(self, key)[begin:end]
                else:
                    hyperparameters[key] = []  
            t = threading.Thread(target=self.search, args=(hyperparameters, results))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return results

