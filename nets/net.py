import torch
import torch.nn as nn



class NyNetwork(nn.Module):
    def __init__(self):
        super(NyNetwork, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.Dropout(p=0.5),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.Dropout(p=0.5),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.SiLU(),
            nn.Linear(16, 4)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4, 8),
            nn.SiLU(),
            nn.Linear(8, 4)
        )
        self.sfc = nn.Sequential(
            nn.Linear(8, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )
    def forward(self, tensor1, tensor2):
        tensor1 = self.fc1(tensor1)
        tensor2 = self.fc2(tensor2)
        tensor3 = torch.cat((tensor1, tensor2), dim=1)
        output = self.sfc(tensor3)
        return output

