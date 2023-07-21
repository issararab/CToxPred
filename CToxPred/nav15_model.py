import torch
import torch.nn.functional as F

class Nav15Classifier(torch.nn.Module):
    """
    Nav1.5 Classifier architecture to process fingerprints and descriptors.

    Parameters:
        inputSize : int
            The input size of the descriptors and fingerprints representations.
        output_dim : int 
            The output dimension of the classifier.

    """
    def __init__(self, inputSize, outputSize):
        super(Nav15Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(inputSize, 200, bias=True)
        torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        self.bn1 = torch.nn.BatchNorm1d(num_features=200)
        
        self.linear2 = torch.nn.Linear(200, 200, bias=True)
        torch.nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
        self.bn2 = torch.nn.BatchNorm1d(num_features=200)
        
        self.linear3 = torch.nn.Linear(200, outputSize, bias=True)
        torch.nn.init.kaiming_normal_(self.linear3.weight, nonlinearity='relu')
        
    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
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