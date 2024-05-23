"""Train a ChemGCN model."""

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

#### Start by creating dataset
main_path = os.getcwd()
data_path = os.path.join(main_path, "data", "solubility_data.csv")
data = pd.read_csv(data_path)
data["carbon_number"] = data["smiles"].apply(lambda x: len(re.findall("C", x)))
data["solubility"] = data["measured log solubility in mols per litre"]
# Define property to predict
property_to_predict = "solubility"

if property_to_predict == "carbon_number":
    labels = data['carbon_number'].values
else:
    labels = data['solubility'].values

# Save new data
new_data_path = os.path.join(main_path, "data", "solubility_carbon.csv")
data.to_csv(new_data_path, index=False)
data_path = os.path.join(main_path, "data", "solubility_carbon.csv")

dataset = GraphData(
    dataset_path=data_path,
    max_atoms=max_atoms,
    node_vec_len=node_vec_len,
    property_name=property_to_predict
)

#### Split data into training and test sets
# Get train and test sizes
dataset_indices = np.arange(0, len(dataset), 1)
train_size = int(np.round(train_size * len(dataset)))
test_size = len(dataset) - train_size

# Randomly sample train and test indices
train_indices = np.random.choice(dataset_indices, size=train_size, replace=False)
test_indices = np.array(list(set(dataset_indices) - set(train_indices)))

# Create dataoaders
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    collate_fn=collate_graph_dataset,
)
test_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=test_sampler,
    collate_fn=collate_graph_dataset,
)

#### Initialize model, standardizer, optimizer, and loss function
# Model
model = ChemGCN(
    node_vec_len=node_vec_len,
    node_fea_len=hidden_nodes,
    hidden_fea_len=hidden_nodes,
    n_conv=n_conv_layers,
    n_hidden=n_hidden_layers,
    n_outputs=1,
    p_dropout=0.1,
)
# Transfer to GPU if needed
if use_GPU:
    model.cuda()

# Standardizer
outputs = [dataset[i][1] for i in range(len(dataset))]
standardizer = Standardizer(torch.Tensor(outputs))

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Loss function
loss_fn = torch.nn.MSELoss()

#### Train the model
loss = []
mae = []
epoch = []
for i in range(n_epochs):
    epoch_loss, epoch_mae = train_model(
        i,
        model,
        train_loader,
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

#### Test the model
# Call test model function
test_loss, test_mae = test_model(
    model, test_loader, loss_fn, standardizer, use_GPU, max_atoms, node_vec_len
)

#### Print final results
print(f"Training Loss: {loss[-1]:.2f}")
print(f"Training MAE: {mae[-1]:.2f}")
print(f"Test Loss: {test_loss:.2f}")
print(f"Test MAE: {test_mae:.2f}")

#### Plots
os.path.join(main_path, "data", "solubility_data.csv")
plot_dir = os.path.join(main_path,"plots")
parity_plot(
    plot_dir, model, test_loader, standardizer, use_GPU, max_atoms, node_vec_len, property_to_predict
)
loss_curve(plot_dir, epoch, loss, property_to_predict)
