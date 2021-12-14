import torch
import torch.nn as nn
import torch.optim as optim
from ray import train
from ray.train import Trainer

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
import ipdb
import time

file = 'covtype.libsvm.binary'
X_train, y_train = load_svmlight_file(file)
X_train = np.array(X_train.todense(), dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
# X_train = X_train[0:10000, :]
# y_train = y_train[0:10000]


X_train = pd.read_csv('housing.csv', header=None)
X_train = np.array(X_train, dtype=np.float32)

#ipdb.set_trace()



# In this example we use a randomly generated dataset.


num_samples = 506
input_size = 14
layer_size = 10
output_size = 1

input = torch.from_numpy(X_train)
#labels = torch.from_numpy(y_train)
labels = torch.randn(num_samples, output_size)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, layer_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(layer_size, output_size)

    def forward(self, input):
        return self.layer2(self.relu(self.layer1(input)))



def train_func():
    num_epochs = 100
    model = NeuralNetwork()
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        output = model(input)
        # import ipdb
        # ipdb.set_trace()
        loss = loss_fn(output.flatten(), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch}, loss: {loss.item()}")


def train_func_distributed():
    num_epochs = 6
    model = NeuralNetwork()
    model = train.torch.prepare_model(model)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        output = model(input)
        loss = loss_fn(output.flatten(), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch}, loss: {loss.item()}")

# train_func()

trainer = Trainer(backend="torch", num_workers=1)
trainer.start()

start_time = time.time()
results = trainer.run(train_func_distributed)
print("--- %s seconds ---" % (time.time() - start_time))

trainer.shutdown()
