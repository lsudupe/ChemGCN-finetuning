"""Train a ChemGCN model with multiple features and predict one via finetuning."""

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
n_conv_layers = 2
n_hidden_layers = 2
learning_rate = 0.01
n_epochs = 30


