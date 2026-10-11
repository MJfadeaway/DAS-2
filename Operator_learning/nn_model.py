from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch 
import torch.nn as nn 
import math 


class FCNN(torch.nn.Module):
    def __init__(self, layers):
        super(FCNN, self).__init__()
        self.layers = layers
        self.layers_hid_num = len(layers)-1

        fc = []
        for i in range(self.layers_hid_num):
            fc.append(torch.nn.Linear(self.layers[i],self.layers[i+1]))
        self.fc = torch.nn.Sequential(*fc)
    
    def forward(self, x):
        for i in range(self.layers_hid_num-1):
            x = torch.tanh(self.fc[i](x))
        
        return self.fc[-1](x)
    

class DeepONet(torch.nn.Module):
    def __init__(self, branch_layers, trunk_layers):
        super(DeepONet, self).__init__()
        assert branch_layers[-1] == trunk_layers[-1]

        self.branch_net = FCNN(branch_layers)
        self.trunk_net = FCNN(trunk_layers)

        self.bias = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, x, sensor_values):
        embedded_x = self.trunk_net(x)
        embedded_sensor_values = self.branch_net(sensor_values) 

        dotted_value = torch.sum(embedded_x * embedded_sensor_values, dim=1, keepdim=True)

        return x * (dotted_value + self.bias) 
    



"""
This code is modified from neural function distributions (https://github.com/EmilienDupont/neural-function-distributions)
"""


class FunctionRepresentation(nn.Module):
    """Function to represent a matrix M. For example this could be a
    function that takes indexed coordinates as input and returns its entry values, i.e.
    f(x(i), y(j)) = M(i, j).

    Args:
    -----
        coordinate_dim (int): Dimension of input (coordinates).
        feature_dim (int): Dimension of output (features).
        layer_sizes (tuple of ints): Specifies size of each hidden layer.
        encoding (torch.nn.Module): Encoding layer, usually one of
            Identity or FourierFeatures.
        final_non_linearity (torch.nn.Module): Final non linearity to use.
            Usually nn.Sigmoid() or nn.Tanh().
    """
    def __init__(self, coordinate_dim, feature_dim, layer_sizes, encoding,
                 non_linearity=nn.ReLU(), final_non_linearity=nn.Tanh()):
        super(FunctionRepresentation, self).__init__()
        self.coordinate_dim = coordinate_dim
        self.feature_dim = feature_dim
        self.layer_sizes = layer_sizes
        self.encoding = encoding
        self.non_linearity = non_linearity
        self.final_non_linearity = final_non_linearity

        self._init_neural_net()

    def _init_neural_net(self):
        """
        """
        # First layer transforms coordinates into a positional encoding
        # Check output dimension of positional encoding
        if isinstance(self.encoding, nn.Identity):
            prev_num_units = self.coordinate_dim  # No encoding, so same output dimension
        else:
            prev_num_units = self.encoding.feature_dim
        # Build MLP layers
        forward_layers = []
        for num_units in self.layer_sizes:
            forward_layers.append(nn.Linear(prev_num_units, num_units))
            forward_layers.append(self.non_linearity)
            prev_num_units = num_units
        forward_layers.append(nn.Linear(prev_num_units, self.feature_dim))
        forward_layers.append(self.final_non_linearity)
        self.forward_layers = nn.Sequential(*forward_layers)

    def forward(self, coordinates):
        """Forward pass. Given a set of coordinates, compute outputs

        Args:
        -----
            coordinates (torch.Tensor): Shape (batch_size, coordinate_dim)

        Returns:
        --------
            returns feature at every coordinate.
        """
        encoded = self.encoding(coordinates)
        return self.forward_layers(encoded)

    def get_weight_shapes(self):
        """Returns lists of shapes of weights and biases in the network."""
        weight_shapes = []
        bias_shapes = []
        for param in self.forward_layers.parameters():
            if len(param.shape) == 1:
                bias_shapes.append(param.shape)
            if len(param.shape) == 2:
                weight_shapes.append(param.shape)
        return weight_shapes, bias_shapes

    def get_weights_and_biases(self):
        """Returns list of weights and biases in the network."""
        weights = []
        biases = []
        for param in self.forward_layers.parameters():
            if len(param.shape) == 1:
                biases.append(param)
            if len(param.shape) == 2:
                weights.append(param)
        return weights, biases

    def set_weights_and_biases(self, weights, biases):
        """Sets weights and biases in the network.

        Args:
        -----
            weights (list of torch.Tensor):
            biases (list of torch.Tensor):

        Notes:
            The inputs to this function should have the same form as the outputs
            of self.get_weights_and_biases.
        """
        weight_idx = 0
        bias_idx = 0
        with torch.no_grad():
            for param in self.forward_layers.parameters():
                if len(param.shape) == 1:
                    param.copy_(biases[bias_idx])
                    bias_idx += 1
                if len(param.shape) == 2:
                    param.copy_(weights[weight_idx])
                    weight_idx += 1

    def _get_config(self):
        return {"coordinate_dim": self.coordinate_dim,
                "feature_dim": self.feature_dim,
                "layer_sizes": self.layer_sizes,
                "encoding": self.encoding,
                "non_linearity": self.non_linearity,
                "final_non_linearity": self.final_non_linearity}


class FourierFeatures(nn.Module):
    """Random Fourier features.

    Args:
    -----
        frequency_matrix (torch.Tensor): Matrix of frequencies to use
            for Fourier features. Shape (num_frequencies, num_coordinates).
            This is referred to as B in the paper.
        learnable_features (bool): If True, fourier features are learnable,
            otherwise they are fixed.
    """
    def __init__(self, frequency_matrix, learnable_features=False):
        super(FourierFeatures, self).__init__()
        if learnable_features:
            self.frequency_matrix = nn.Parameter(frequency_matrix)
        else:
            # Register buffer adds a key to the state dict of the model. This will
            # track the attribute without registering it as a learnable parameter.
            # We require this so frequency matrix will also be moved to GPU when
            # we call .to(device) on the model
            self.register_buffer('frequency_matrix', frequency_matrix)
        self.learnable_features = learnable_features
        self.num_frequencies = frequency_matrix.shape[0]
        self.coordinate_dim = frequency_matrix.shape[1]
        # Factor of 2 since we consider both a sine and cosine encoding
        self.feature_dim = 2 * self.num_frequencies

    def forward(self, coordinates):
        """Creates Fourier features from coordinates.

        Args:
        -----
            coordinates (torch.Tensor): Shape (num_points, coordinate_dim)
        """
        # The coordinates variable contains a batch of vectors of dimension
        # coordinate_dim. We want to perform a matrix multiply of each of these
        # vectors with the frequency matrix. I.e. given coordinates of
        # shape (num_points, coordinate_dim) we perform a matrix multiply by
        # the transposed frequency matrix of shape (coordinate_dim, num_frequencies)
        # to obtain an output of shape (num_points, num_frequencies).
        prefeatures = torch.matmul(coordinates, self.frequency_matrix.T)
        # Calculate cosine and sine features
        cos_features = torch.cos(2 * math.pi * prefeatures)
        sin_features = torch.sin(2 * math.pi * prefeatures)
        # Concatenate sine and cosine features
        return torch.cat((cos_features, sin_features), dim=1)


