"""Utility functions for multiplle property."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error


class Standardizer:
    def __init__(self, X):
        """
        Class to standardize ChemGCN outputs

        Parameters
        ----------
        X : torch.Tensor
            Tensor of outputs
        """
        self.mean = torch.mean(X)
        self.std = torch.std(X)

    def standardize(self, X):
        """
        Convert a non-standardized output to a standardized output

        Parameters
        ----------
        X : torch.Tensor
            Tensor of non-standardized outputs

        Returns
        -------
        Z : torch.Tensor
            Tensor of standardized outputs

        """
        Z = (X - self.mean) / (self.std)
        return Z

    def restore(self, Z):
        """
        Restore a standardized output to the non-standardized output

        Parameters
        ----------
        Z : torch.Tensor
            Tensor of standardized outputs

        Returns
        -------
        X : torch.Tensor
            Tensor of non-standardized outputs

        """
        X = self.mean + Z * self.std
        return X

    def state(self):
        """
        Return dictionary of the state of the Standardizer

        Returns
        -------
        dict
            Dictionary with the mean and std of the outputs

        """
        return {"mean": self.mean, "std": self.std}

    def load(self, state):
        """
        Load a dictionary containing the state of the Standardizer and assign mean and std

        Parameters
        ----------
        state : dict
            Dictionary containing mean and std
        """
        self.mean = state["mean"]
        self.std = state["std"]


# Utility functions to train, test model
def train_model(
    epoch,
    model,
    training_dataloader,
    optimizer,
    loss_fn,
    standardizers,
    use_GPU,
    max_atoms,
    node_vec_len,
):
    """
    Execute training of one epoch for the ChemGCN model.

    Parameters
    ----------
    epoch : int
        Current epoch
    model : ChemGCN
        ChemGCN model object
    training_dataloader : data.DataLoader
        Training DataLoader
    optimizer : torch.optim.Optimizer
        Model optimizer
    loss_fn : dict
        Dictionary of loss functions for each property
    standardizers : dict
        Dictionary of Standardizers for each property
    use_GPU: bool
        Whether to use GPU
    max_atoms: int
        Maximum number of atoms in graph
    node_vec_len: int
        Maximum node vector length in graph

    Returns
    -------
    avg_loss : dict
        Training loss averaged over batches for each property
    avg_mae : dict
        Training MAE averaged over batches for each property
    """

    avg_loss = {prop: 0 for prop in loss_fn.keys()}  # for multiple properties
    avg_mae = {prop: 0 for prop in loss_fn.keys()}  # for multiple properties
    count = 0

    model.train()

    for i, dataset in enumerate(training_dataloader):
        node_mat, adj_mat = dataset[0]
        output = dataset[1]

        first_dim = int((torch.numel(node_mat)) / (max_atoms * node_vec_len))
        node_mat = node_mat.reshape(first_dim, max_atoms, node_vec_len)
        adj_mat = adj_mat.reshape(first_dim, max_atoms, max_atoms)

        if use_GPU:
            node_mat, adj_mat = node_mat.cuda(), adj_mat.cuda()

        nn_input = (node_mat, adj_mat)

        optimizer.zero_grad()

        nn_prediction = model(*nn_input)

        loss = 0
        for prop in loss_fn.keys():
            output_std = standardizers[prop].standardize(output[:, properties.index(prop)].view(-1, 1))
            if use_GPU:
                output_std = output_std.cuda()
            loss += loss_fn[prop](nn_prediction[prop], output_std)
            avg_loss[prop] += loss_fn[prop](nn_prediction[prop], output_std).item()

            prediction = standardizers[prop].restore(nn_prediction[prop].detach().cpu())
            mae = mean_absolute_error(output[:, properties.index(prop)].view(-1, 1), prediction)
            avg_mae[prop] += mae

        loss.backward()
        optimizer.step()

        count += 1

    avg_loss = {prop: avg_loss[prop] / count for prop in loss_fn.keys()}
    avg_mae = {prop: avg_mae[prop] / count for prop in loss_fn.keys()}

    print(f"Epoch: [{epoch}]\t" + "\t".join([f"{prop} Loss: [{avg_loss[prop]:.2f}] MAE: [{avg_mae[prop]:.2f}]" for prop in loss_fn.keys()]))

    return avg_loss, avg_mae


def test_model(
    model,
    test_dataloader,
    loss_fn,
    standardizers,
    use_GPU,
    max_atoms,
    node_vec_len,
):
    """
    Test the ChemGCN model.

    Parameters
    ----------
    model : ChemGCN
        ChemGCN model object
    test_dataloader : data.DataLoader
        Test DataLoader
    loss_fn : dict
        Dictionary of loss functions for each property
    standardizers : dict
        Dictionary of Standardizers for each property
    use_GPU: bool
        Whether to use GPU
    max_atoms: int
        Maximum number of atoms in graph
    node_vec_len: int
        Maximum node vector length in graph

    Returns
    -------
    test_loss : dict
        Test loss for each property
    test_mae : dict
        Test MAE for each property
    """

    test_loss = {prop: 0 for prop in loss_fn.keys()}
    test_mae = {prop: 0 for prop in loss_fn.keys()}
    count = 0

    model.eval()

    with torch.no_grad():
        for i, dataset in enumerate(test_dataloader):
            node_mat, adj_mat = dataset[0]
            output = dataset[1]

            first_dim = int((torch.numel(node_mat)) / (max_atoms * node_vec_len))
            node_mat = node_mat.reshape(first_dim, max_atoms, node_vec_len)
            adj_mat = adj_mat.reshape(first_dim, max_atoms, max_atoms)

            if use_GPU:
                node_mat, adj_mat = node_mat.cuda(), adj_mat.cuda()

            nn_input = (node_mat, adj_mat)

            nn_prediction = model(*nn_input)

            for prop in loss_fn.keys():
                output_std = standardizers[prop].standardize(output[:, properties.index(prop)].view(-1, 1))
                if use_GPU:
                    output_std = output_std.cuda()
                test_loss[prop] += loss_fn[prop](nn_prediction[prop], output_std).item()

                prediction = standardizers[prop].restore(nn_prediction[prop].detach().cpu())
                mae = mean_absolute_error(output[:, properties.index(prop)].view(-1, 1), prediction)
                test_mae[prop] += mae

            count += 1

    test_loss = {prop: test_loss[prop] / count for prop in loss_fn.keys()}
    test_mae = {prop: test_mae[prop] / count for prop in loss_fn.keys()}

    return test_loss, test_mae

def parity_plot(
    save_dir,
    model,
    test_dataloader,
    standardizer,
    use_GPU,
    max_atoms,
    node_vec_len,
    property_name
):
    """
    Create a parity plot for the ChemGCN model.

    Parameters
    ----------
    save_dir: str
        Name of directory to store the parity plot in
    model : ChemGCN
        ChemGCN model object
    test_dataloader : data.DataLoader
        Test DataLoader
    standardizer : Standardizer
        Standardizer object
    use_GPU: bool
        Whether to use GPU
    max_atoms: int
        Maximum number of atoms in graph
    node_vec_len: int
        Maximum node vector length in graph
    property_name: str
        name of the property to predict
    """

    # Create variables to store losses and error
    outputs = []
    predictions = []

    # Switch model to train mode
    model.eval()

    # Go over each batch in the dataloader
    for i, dataset in enumerate(test_dataloader):
        # Unpack data
        node_mat = dataset[0][0]
        adj_mat = dataset[0][1]
        output = dataset[1]

        # Reshape inputs
        first_dim = int((torch.numel(node_mat)) / (max_atoms * node_vec_len))
        node_mat = node_mat.reshape(first_dim, max_atoms, node_vec_len)
        adj_mat = adj_mat.reshape(first_dim, max_atoms, max_atoms)

        # Package inputs and outputs; check if GPU is enabled
        if use_GPU:
            nn_input = (node_mat.cuda(), adj_mat.cuda())
        else:
            nn_input = (node_mat, adj_mat)

        # Compute output from network
        nn_prediction = model(*nn_input)

        # Calculate MAE
        prediction = standardizer.restore(nn_prediction.detach().cpu())

        # Add to list
        outputs.append(output)
        predictions.append(prediction)

    # Flatten
    outputs_arr = np.concatenate(outputs)
    preds_arr = np.concatenate(predictions)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=500)
    ax.scatter(
        outputs_arr, preds_arr, marker="o", color="mediumseagreen", edgecolor="black"
    )

    min_plot = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_plot = max(ax.get_xlim()[1], ax.get_ylim()[1])
    min_plot = (1 - np.sign(min_plot) * 0.2) * min_plot
    max_plot = (1 + np.sign(max_plot) * 0.2) * max_plot

    ax.plot([min_plot, max_plot], [min_plot, max_plot], linestyle="-", color="black")
    ax.margins(x=0, y=0)
    ax.set_xlim([min_plot, max_plot])
    ax.set_ylim([min_plot, max_plot])
    ax.set_xlabel(f"Measured values {property_name}")
    ax.set_ylabel(f"ChemGCN predictions {property_name}")
    ax.set_title(f"Parity plot for {property_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"parity_plot_{property_name}.png"))


def loss_curve(save_dir, epochs, losses, property_name):
    """
    Make a loss curve.

    Parameters
    ----------
    save_dir: str
        Name of directory to store plot in
    epochs: list
        List of epochs
    losses: list
        List of losses
    property_name
        name of the property to predict

    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=500)
    ax.plot(epochs, losses, marker="o", linestyle="--", color="royalblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared loss")
    ax.set_title(f"Loss curve for {property_name} prediction")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"loss_curve_{property_name}.png"))
