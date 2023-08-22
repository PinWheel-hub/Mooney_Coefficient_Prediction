import torch
from torch.utils.data.dataset import Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import json
import numpy as np
import torch.nn as nn


class NyNetwork2(nn.Module):
    def __init__(self):
        super(NyNetwork2, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
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
        print(tensor1.shape)
        tensor2 = self.fc2(tensor2)
        tensor3 = torch.cat((tensor1, tensor2), dim=1)
        #output = self.sfc(tensor3)
        return tensor3

class TestDataset(Dataset):
    def __init__(self, annotation_lines):
        super(TestDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        with open(self.annotation_lines[index], 'r') as f:
            line = f.readline()
            data = json.loads(line)
            seconds = data['seconds']
            energy = data['energy']
            temp = data['temp']
            speed = data['speed']
            power = data['power']
            press = data['press']
            type1 = data['type1']
            type2 = data['type2']
            seconds = torch.tensor([seconds])
            energy = torch.tensor([energy])
            temp = torch.tensor(temp)
            speed = torch.tensor(speed)
            power = torch.tensor(power)
            press = torch.tensor(press)
            type1 = torch.tensor([type1])
            type2 = torch.tensor([type2])
            tensor1 = torch.cat((temp, speed, power, press), dim=0).reshape(1, 4, 10, 10)
            tensor2 = torch.cat((seconds, energy, type1, type2), dim=0).reshape(1, 4)
        return tensor1, tensor2

if __name__ == "__main__":
    model = NyNetwork2()
    model_path = 'model_data/ep039-loss17.196-val_loss23.579.pth'
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    model = model.eval()
    # parameters = MyModel.parameters()
    # for p in parameters:
    #     print(p)

    test_annotation_path = 'my_dataset_change/val.txt'
    with open(test_annotation_path, encoding='utf-8') as f:
        test_lines = f.read().splitlines()
    test_dataset = TestDataset(test_lines)
    #with open("test_result.txt", "w") as f:
    for i in range(0, len(test_lines)):
        output = model(test_dataset[i][0], test_dataset[i][1])
        #f.write(str(float(output.item())) + '\n')
        #print(output)