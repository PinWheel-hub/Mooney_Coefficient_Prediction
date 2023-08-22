from random import sample, shuffle
import torch
from torch.utils.data.dataset import Dataset
import json
class MenniDataset(Dataset):
    def __init__(self, json_file, epoch_length, train):
        super(MenniDataset, self).__init__()
        with open(json_file, 'r') as f:
            self.datas = f.read().splitlines()
        self.length = len(self.datas)
        self.epoch_length       = epoch_length
        self.train              = train
        self.epoch_now          = -1

    def __len__(self):
        return self.length
    
    def change_length(self, l, length):
        new_l = []
        for i in range(0, length):
            new_l.append(l[int(i * len(l) / length)])
        return new_l

    def __getitem__(self, index):
        index = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        data = json.loads(self.datas[index])
        seconds = data['seconds']
        energy = data['energy']
        temp = data['temp']
        temp = self.change_length(temp, 100)
        speed = data['speed']
        speed = self.change_length(speed, 100)
        power = data['power']
        power = self.change_length(power, 100)
        press = data['press']
        press = self.change_length(press, 100)
        type1 = data['type1']
        type2 = data['type2']
        target = data['target']
        seconds = torch.tensor([seconds])
        energy = torch.tensor([energy])
        temp = torch.tensor(temp)
        speed = torch.tensor(speed)
        power = torch.tensor(power)
        press = torch.tensor(press)
        type1 = torch.tensor([type1])
        type2 = torch.tensor([type2])
        target = torch.tensor([target])
        tensor1 = torch.cat((temp, speed, power, press), dim=0).reshape(4, 10, 10)
        tensor2 = torch.cat((seconds, energy, type1, type2), dim=0).reshape(4)
        target = target.reshape(1)
        return tensor1, tensor2, target

# with open('my_dataset_change/train.txt', encoding='utf-8') as f:
#     train_lines = f.read().splitlines()
# trainset = MenniDataset(train_lines, 0, True)
# print(trainset[1][1])


# DataLoader中collate_fn使用
# def dataset_collate(batch):
#     tensors1 = []
#     tensors2 = []
#     targets = []
#     for tensor1, tensor2, target in batch:
#         tensors1.append(tensor1)
#         tensors2.append(tensor2)
#         targets.append(target)
#     return tensors1, tensors2, targets