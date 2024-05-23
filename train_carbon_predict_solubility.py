"""Train a ChemGCN model with carbon information and check solubility predictions."""

import numpy as np
import torch
import os
import pandas as pd
import re


from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from chem_gcn.model import ChemGCN
from chem_gcn.utils import (
    train_model,
    test_model,
    parity_plot,
    loss_curve,
    Standardizer,
)
from chem_gcn.graphs import GraphData, collate_graph_dataset


#### Fix seeds
np.random.seed(0)
torch.manual_seed(0)
use_GPU = torch.cuda.is_available()

#### Prepare data
path = os.getcwd()
data_path = os.path.join(path, "data", "solubility_carbon.csv")
data = pd.read_csv(data_path)

#### Inputs
max_atoms = 200
node_vec_len = 60
train_size = 0.7
batch_size = 32
hidden_nodes = 60
n_conv_layers = 4
n_hidden_layers = 2
learning_rate = 0.01
n_epochs = 30


#### STEP 1: Train the model for the number of carbons
## Prepare data
dataset_carbon = GraphData(
    dataset_path=data_path,
    max_atoms= max_atoms,
    node_vec_len= node_vec_len,
    property_name="carbon_number"
)

## Conf the model
model_carbon = ChemGCN(
    node_vec_len= node_vec_len,
    node_fea_len= hidden_nodes,
    hidden_fea_len= hidden_nodes,
    n_conv= n_conv_layers,
    n_hidden= n_hidden_layers,
    n_outputs=1
)

## Split data into training and test sets
# Get train and test sizes
dataset_indices = np.arange(0, len(dataset_carbon), 1)
train_size = int(np.round(train_size * len(dataset_carbon)))
test_size = len(dataset_carbon) - train_size

# Randomly sample train and test indices
train_indices = np.random.choice(dataset_indices, size=train_size, replace=False)
test_indices = np.array(list(set(dataset_indices) - set(train_indices)))

# Create dataoaders
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader_carbon = DataLoader(
    dataset_carbon,
    batch_size=batch_size,
    sampler=train_sampler,
    collate_fn=collate_graph_dataset,
)
test_loader_carbon = DataLoader(
    dataset_carbon,
    batch_size=batch_size,
    sampler=test_sampler,
    collate_fn=collate_graph_dataset,
)

## Train the model
# Optimizer
optimizer = torch.optim.Adam(model_carbon.parameters(), lr=learning_rate)
# Loss function
loss_fn = torch.nn.MSELoss()
# Standardizer
outputs = [dataset_carbon[i][1] for i in range(len(dataset_carbon))]
standardizer = Standardizer(torch.Tensor(outputs))

loss = []
mae = []
epoch = []
for i in range(n_epochs):
    epoch_loss, epoch_mae = train_model(
        i,
        model_carbon,
        train_loader_carbon,
        optimizer,
        loss_fn,
        standardizer,
        use_GPU,
        max_atoms,
        node_vec_len,
    )
    loss.append(epoch_loss)
    mae.append(epoch_mae)
    epoch.append(i)


#### STEP 2: Evaluate the impact in Solubility prediction

## 2.1 Reuse convolutional layers
# Create new instancy for solubility
model_solubility = ChemGCN(
    node_vec_len=node_vec_len,
    node_fea_len=hidden_nodes,
    hidden_fea_len=hidden_nodes,
    n_conv=n_conv_layers,
    n_hidden=n_hidden_layers,
    n_outputs=1  # Salida para solubilidad
)

# Copy the convolutional layers used for the carbon model
for i in range(n_conv_layers):
    model_solubility.conv_layers[i].load_state_dict(model_carbon.conv_layers[i].state_dict())

## 2.2 Retrain the dense layers
# Prepare data for solubility
dataset_solubility = GraphData(
    dataset_path=data_path,
    max_atoms= max_atoms,
    node_vec_len= node_vec_len,
    property_name="solubility"
)

# Get train and test sizes
dataset_indices = np.arange(0, len(dataset_solubility), 1)
train_ratio = 0.7
train_size = int(np.round(train_ratio * len(dataset_solubility)))
test_size = len(dataset_solubility) - train_size

# Randomly sample train and test indices
train_indices = np.random.choice(dataset_indices, size=train_size, replace=False)
test_indices = np.array(list(set(dataset_indices) - set(train_indices)))

# Create dataoaders
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader_solubility = DataLoader(
    dataset_solubility,
    batch_size=batch_size,
    sampler=train_sampler,
    collate_fn=collate_graph_dataset,
)
test_loader_solubility = DataLoader(
    dataset_solubility,
    batch_size=batch_size,
    sampler=test_sampler,
    collate_fn=collate_graph_dataset,
)

## 2.3 Retrain only the dense layer and output layer using solubility data
# Asegúrate de congelar las capas convolucionales para no reentrenarlas
for param in model_solubility.conv_layers.parameters():
    param.requires_grad = False

# Entrenamiento solo en las capas densas y de salida
# Optimizer
optimizer_solubility = torch.optim.Adam(model_solubility.parameters(),
                                        lr=learning_rate)
# Loss function
loss_fn_solubility = torch.nn.MSELoss()
# Standardizer
outputs_solubility = [dataset_solubility[i][1] for i in range(len(dataset_solubility))]
standardizer_solubility = Standardizer(torch.Tensor(outputs_solubility))

loss_solubility = []
mae_solubility = []
epoch_solubility = []
for i in range(n_epochs):
    epoch_loss, epoch_mae = train_model(
        i,
        model_solubility,
        train_loader_solubility,
        optimizer_solubility,
        loss_fn_solubility,
        standardizer_solubility,
        use_GPU,
        max_atoms,
        node_vec_len,
    )
    loss_solubility.append(epoch_loss)
    mae_solubility.append(epoch_mae)
    epoch_solubility.append(i)


#### STEP 3: Evaluate and analysis
#Evalúa cómo el modelo reentrenado predice la solubilidad utilizando el
# conjunto de datos de prueba y compara los resultados con los obtenidos
# por el modelo originalmente entrenado para solubilidad.
# Esto te dará una idea de cómo el aprendizaje previo del
# número de carbonos puede influir en la tarea de predecir la solubilidad.

test_loss, test_mae = test_model(
    model_solubility, test_loader_solubility, loss_fn, standardizer, use_GPU, max_atoms, node_vec_len
)

#### Print final results
print(f"Training Loss: {loss[-1]:.2f}")
print(f"Training MAE: {mae[-1]:.2f}")
print(f"Test Loss: {test_loss:.2f}")
print(f"Test MAE: {test_mae:.2f}")


#### Plots
main_path = os.getcwd()
os.path.join(main_path, "data", "solubility_data.csv")
plot_dir = os.path.join(main_path,"plots")
property_to_predict = "solubility"
parity_plot(
    plot_dir, model_solubility, test_loader_solubility,
    standardizer, use_GPU, max_atoms, node_vec_len, property_to_predict
)
loss_curve(plot_dir, epoch_solubility, loss_solubility, property_to_predict)

