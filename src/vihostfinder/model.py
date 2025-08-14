import torch.nn as nn
import torch.nn.functional as F


class MultiLabelFFNN(nn.Module):
    
    def __init__(self, input_dim=256, hidden_dim=384, hidden_dim2=128,
                    output_dim=13, dropout_rate=0.0):
    
        super(MultiLabelFFNN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # No sigmoid here if using BCEWithLogitsLoss
        x = self.dropout(x)
        x = self.fc3(x)
        return x