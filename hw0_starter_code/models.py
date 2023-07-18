"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """
    def reset_parameters(self):
        """
        Initialize parameters.
        """
        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        # Initialize the embedding layer with values sampled from a normal distribution
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------
    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.
    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 sparse=False, embedding_sharing=True):

        super().__init__()
        self.embedding_dim = embedding_dim
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        self.embedding_sharing = embedding_sharing
        self.num_users = num_users
        self.num_items = num_items
        self.U = ScaledEmbedding(self.num_users, embedding_dim)
        self.Q = ScaledEmbedding(self.num_items, embedding_dim)
        self.A = ZeroEmbedding(self.num_users,1)
        self.B = ZeroEmbedding(self.num_items, 1)


        if self.embedding_sharing:
            self.UR = self.U
            self.QR = self.Q

        else :
            self.UR = ScaledEmbedding(self.num_users,embedding_dim)
            self.QR = ScaledEmbedding(self.num_users,embedding_dim)
        
        self.mlp_matrix_factorization= self.create_mlp(layer_sizes)
        self.mlp_regression = self.create_mlp(layer_sizes)

    def create_mlp(self, layer_sizes):
        layers = []
        in_dim = self.embedding_dim * 3  # Concatenated input size is 96 (32 * 3)

        for out_dim in layer_sizes:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, 1))  # Final output layer with output size 1
        return nn.Sequential(*layers)
        #********************************************************
        #********************************************************
        #********************************************************

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        ui = self.U(user_ids)
        qj = self.Q(item_ids)
        ai = self.A(user_ids)
        bj = self.B(item_ids)
        pij = (ui*qj).sum(dim = 1)+ ai.squeeze()+bj.squeeze()
        if self.embedding_sharing:
            rij = self.mlp_matrix_factorization(torch.cat((ui,qj,ui*qj),dim = 1))
        else : 
            ri = self.UR(user_ids)
            pj = self.QR(item_ids)
            rij = self.mlp_regression(torch.cat((ri,pj,ri*pj),dim = 1))

        predictions = torch.sigmoid(pij)
        score = rij.squeeze()
        #********************************************************
        #********************************************************
        #********************************************************
        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")
        
        return predictions, score