import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


class ClassifierFramework(torch.nn.Module):
    """
    Classifier framework to process various molecular feature types, 
    including fingerprints, descriptors, and graph representations.

    Parameters:
        descriptors_input_dim : int
            The input size of the descriptors and/or fingerprints path.
        graph_nodes_max_dim : int 
            The maximum number of atoms in the molecular structure (graph path).
        output_dim : int 
            The output dimension of the classifier.

    """
    def __init__(self, descriptors_input_dim: int, graph_nodes_max_dim: int,
                 output_dim: int):
        super(ClassifierFramework, self).__init__()

        ## Graph processing path
        # Define the first GCNConv layer and BatchNormalization
        self.conv1 = GCNConv(67, 64)
        self.batch_norm1 = torch.nn.BatchNorm1d(graph_nodes_max_dim)

        # Define the second GCNConv layer and BatchNormalization
        self.conv2 = GCNConv(64, 32)
        self.batch_norm2 = torch.nn.BatchNorm1d(graph_nodes_max_dim)

        # Define the pooling layer
        self.pool = global_max_pool

        ## Fingerprints processing path
        self.linear1 = torch.nn.Linear(descriptors_input_dim, 400, bias=True)
        torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        self.bn1 = torch.nn.BatchNorm1d(num_features=400)
        self.linear2 = torch.nn.Linear(400, 64, bias=True)
        torch.nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
        self.bn2 = torch.nn.BatchNorm1d(num_features=64)

        # Define the final fully connected layer
        self.fc = torch.nn.Linear(96, output_dim)

    def forward(self, x_fingerprints, x_node_features, edge_list, device):
        ## Graph processing
        # Init required variables
        batch_tensor = torch.arange(x_node_features.shape[0]).view(-1,
                                                                   1).repeat(1,
                                                                             x_node_features.shape[
                                                                                 1])
        batch_tensor = batch_tensor.reshape(-1)
        batch_tensor = batch_tensor.to(device)
        # Start network
        out_graph = self.conv1(x_node_features, edge_list)
        out_graph = self.batch_norm1(out_graph)
        out_graph = F.relu(out_graph)
        # Pass the features through the second GCNConv layer and BatchNormalization
        out_graph = self.conv2(out_graph, edge_list)
        out_graph = self.batch_norm2(out_graph)
        out_graph = F.relu(out_graph)
        # Pooling layer
        x = self.pool(out_graph.reshape(-1, 32), batch_tensor)

        ## Fingerprints
        out_fingerprints = F.relu(self.linear1(x_fingerprints))
        out_fingerprints = F.relu(self.linear2(out_fingerprints))

        ## Merge paths processing
        out = torch.cat((x, out_fingerprints), dim=1)
        # Fully connected layer
        out = self.fc(out)

        return out

    def save(self, path):
        """
        Save model with its parameters to the given path. 
        Conventionally the path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load model dictionary. The
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        device = torch.device('cpu')
        self.load_state_dict(torch.load(path, map_location=device))