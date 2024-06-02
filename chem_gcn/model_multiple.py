"Graph neural network model."

import torch
import torch.nn as nn


#### CLASSES
class ConvolutionLayer(nn.Module):
    """
    Create a simple graph convolution layer
    """

    def __init__(self, node_in_len: int, node_out_len: int):
        super().__init__()
        self.conv_linear = nn.Linear(node_in_len, node_out_len)
        self.conv_activation = nn.LeakyReLU()

    def forward(self, node_mat, adj_mat):
        n_neighbors = adj_mat.sum(dim=-1, keepdims=True)
        self.idx_mat = torch.eye(adj_mat.shape[-2], adj_mat.shape[-1], device=n_neighbors.device)
        idx_mat = self.idx_mat.unsqueeze(0).expand(*adj_mat.shape)
        inv_degree_mat = torch.mul(idx_mat, 1 / n_neighbors)
        node_fea = torch.bmm(inv_degree_mat, adj_mat)
        node_fea = torch.bmm(node_fea, node_mat)
        node_fea = self.conv_linear(node_fea)
        node_fea = self.conv_activation(node_fea)
        return node_fea

class PoolingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, node_fea):
        pooled_node_fea = node_fea.mean(dim=1)
        return pooled_node_fea

class ChemGCN_multiple(nn.Module):
    """
    Create a graph neural network to predict multiple properties of molecules.
    """
    def __init__(self, node_vec_len: int, node_fea_len: int, hidden_fea_len: int, n_conv: int, n_hidden: int, n_outputs: dict, p_dropout: float = 0.0):
        super().__init__()
        self.init_transform = nn.Linear(node_vec_len, node_fea_len)
        self.conv_layers = nn.ModuleList([ConvolutionLayer(node_in_len=node_fea_len, node_out_len=node_fea_len) for _ in range(n_conv)])
        self.pooling = PoolingLayer()
        pooled_node_fea_len = node_fea_len
        self.pooling_activation = nn.LeakyReLU()
        self.pooled_to_hidden = nn.Linear(pooled_node_fea_len, hidden_fea_len)
        self.hidden_layer = nn.Linear(hidden_fea_len, hidden_fea_len)
        self.hidden_activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=p_dropout)
        self.n_hidden = n_hidden
        if self.n_hidden > 1:
            self.hidden_layers = nn.ModuleList([self.hidden_layer for _ in range(n_hidden - 1)])
            self.hidden_activation_layers = nn.ModuleList([self.hidden_activation for _ in range(n_hidden - 1)])
            self.hidden_dropout_layers = nn.ModuleList([self.dropout for _ in range(n_hidden - 1)])
        self.hidden_to_output = nn.ModuleDict({name: nn.Linear(hidden_fea_len, out_dim) for name, out_dim in n_outputs.items()})

    def forward(self, node_mat, adj_mat):
        node_fea = self.init_transform(node_mat)
        for conv in self.conv_layers:
            node_fea = conv(node_fea, adj_mat)
        pooled_node_fea = self.pooling(node_fea)
        pooled_node_fea = self.pooling_activation(pooled_node_fea)
        hidden_node_fea = self.pooled_to_hidden(pooled_node_fea)
        hidden_node_fea = self.hidden_activation(hidden_node_fea)
        hidden_node_fea = self.dropout(hidden_node_fea)
        if self.n_hidden > 1:
            for i in range(self.n_hidden - 1):
                hidden_node_fea = self.hidden_layers[i](hidden_node_fea)
                hidden_node_fea = self.hidden_activation_layers[i](hidden_node_fea)
                hidden_node_fea = self.hidden_dropout_layers[i](hidden_node_fea)
        out = {name: layer(hidden_node_fea) for name, layer in self.hidden_to_output.items()}
        return out
