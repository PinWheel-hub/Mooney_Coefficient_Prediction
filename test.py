import torch
from nets.net import NyNetwork
from torch.utils.data.dataset import Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import json

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
    MyModel = NyNetwork()
    MyModel.load_state_dict(torch.load('model_data/ep026-loss39.217-val_loss23.880.pth', map_location=device))
    MyModel = MyModel.eval()
    # parameters = MyModel.parameters()
    # for p in parameters:
    #     print(p)
    # test_annotation_path = 'my_dataset_change/val.txt'
    # with open(test_annotation_path, encoding='utf-8') as f:
    #     test_lines = f.read().splitlines()
    # test_dataset = TestDataset(test_lines)
    # with open("test_result.txt", "w") as f:
    #     for i in range(0, len(test_lines)):
    #         output = MyModel(test_dataset[i][0], test_dataset[i][1])
    #         f.write(str(float(output.item())) + '\n')
    #         print(str(float(output.item())))
