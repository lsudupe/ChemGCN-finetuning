"""Train a ChemGCN model with multiple features and predict one via finetuning."""

import numpy as np
import torch
import os
import pandas as pd
import re


from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from chem_gcn.model_multiple import ChemGCN_multiple
from chem_gcn.utils_multiple import (
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


# Define properties to predict
properties = ["carbon_number", "solubility"]
n_outputs = {prop: 1 for prop in properties}

# Prepare data
dataset = GraphData(
    dataset_path=data_path,
    max_atoms=max_atoms,
    node_vec_len=node_vec_len,
    property_names=properties
)

# Configure the model
model = ChemGCN_multiple(
    node_vec_len=node_vec_len,
    node_fea_len=hidden_nodes,
    hidden_fea_len=hidden_nodes,
    n_conv=n_conv_layers,
    n_hidden=n_hidden_layers,
    n_outputs=n_outputs
)

# Split data into training and test sets
dataset_indices = np.arange(0, len(dataset), 1)
train_size = int(np.round(train_size * len(dataset)))
test_size = len(dataset) - train_size

# Randomly sample train and test indices
train_indices = np.random.choice(dataset_indices, size=train_size, replace=False)
test_indices = np.array(list(set(dataset_indices) - set(train_indices)))

# Create dataloaders
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

## Step 1: Train the model with both properties

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Loss function
loss_fn = {prop: torch.nn.MSELoss() for prop in properties}
# Standardizer
outputs = {prop: [dataset[i][1][j].item() for i in range(len(dataset))] for j, prop in enumerate(properties)}
standardizers = {prop: Standardizer(torch.Tensor(output)) for prop, output in outputs.items()}

# Training loop
loss = {prop: [] for prop in properties}
mae = {prop: [] for prop in properties}
epoch = []
for i in range(n_epochs):
    epoch_loss, epoch_mae = train_model(
        i,
        model,
        train_loader,
        optimizer,
        loss_fn,
        standardizers,
        use_GPU,
        max_atoms,
        node_vec_len,
    )
    for prop in properties:
        loss[prop].append(epoch_loss[prop])
        mae[prop].append(epoch_mae[prop])
    epoch.append(i)

# Evaluate the model
test_loss, test_mae = test_model(
    model, test_loader, loss_fn, standardizers, use_GPU, max_atoms, node_vec_len
)

# Print final results
for prop in properties:
    print(f"{prop.capitalize()} Training Loss: {loss[prop][-1]:.2f}")
    print(f"{prop.capitalize()} Training MAE: {mae[prop][-1]:.2f}")
    print(f"{prop.capitalize()} Test Loss: {test_loss[prop]:.2f}")
    print(f"{prop.capitalize()} Test MAE: {test_mae[prop]:.2f}")

## Step 2: Froze the conv layers
for param in model.conv_layers.parameters():
    param.requires_grad = False


## Step 3: Retrain the last dense layer
# Fine-tune the model for a specific property (e.g., solubility)
property_to_finetune = "solubility"
fine_tune_optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
)
fine_tune_loss_fn = torch.nn.MSELoss()
fine_tune_standardizer = standardizers[property_to_finetune]

# Training loop for fine-tuning
fine_tune_loss = []
fine_tune_mae = []
fine_tune_epoch = []
for i in range(n_epochs):
    epoch_loss, epoch_mae = train_model(
        i,
        model,
        train_loader,
        fine_tune_optimizer,
        {property_to_finetune: fine_tune_loss_fn},
        {property_to_finetune: fine_tune_standardizer},
        use_GPU,
        max_atoms,
        node_vec_len,
    )
    fine_tune_loss.append(epoch_loss[property_to_finetune])
    fine_tune_mae.append(epoch_mae[property_to_finetune])
    fine_tune_epoch.append(i)

# Evaluate the fine-tuned model
test_loss, test_mae = test_model(
    model, test_loader, {property_to_finetune: fine_tune_loss_fn}, {property_to_finetune: fine_tune_standardizer}, use_GPU, max_atoms, node_vec_len
)

# Print final results for fine-tuning
print(f"Fine-tuned {property_to_finetune.capitalize()} Training Loss: {fine_tune_loss[-1]:.2f}")
print(f"Fine-tuned {property_to_finetune.capitalize()} Training MAE: {fine_tune_mae[-1]:.2f}")
print(f"Fine-tuned {property_to_finetune.capitalize()} Test Loss: {test_loss[property_to_finetune]:.2f}")
print(f"Fine-tuned {property_to_finetune.capitalize()} Test MAE: {test_mae[property_to_finetune]:.2f}")
