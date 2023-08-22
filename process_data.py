import json
import random

def change_length(l, length):
    new_l = []
    for i in range(0, length):
        new_l.append(l[int(i * len(l) / length)])
    return new_l


if __name__ == "__main__":
    random.seed(3)
    index = list(range(200000))
    random.shuffle(index)
    with open("dataset/menni2107train.txt", 'r') as lines, open("dataset/my_train.txt", 'w') as train, open("dataset/my_val.txt", 'w') as val:
        for i, line in enumerate(lines):
            if (index[i] < 160000):
                train.write(line)
            else:
                val.write(line)
            print(i)
    # with open("data/menni2107train.txt", 'r') as lines:
    #     i = 0
    #     for line in lines:
    #         i += 1
    #         if (i <= 160000):
    #             with open("my_dataset_change/train/{0}.txt".format(i), 'w') as f:
    #                 js = json.loads(line)
    #                 temp = js['temp']
    #                 js['temp'] = change_length(temp, 100)
    #                 speed = js['speed']
    #                 js['speed'] = change_length(speed, 100)
    #                 power = js['power']
    #                 js['power'] = change_length(power, 100)
    #                 press = js['press']
    #                 js['press'] = change_length(press, 100)
    #                 f.write(json.dumps(js))
    #         else:
    #             with open("my_dataset_change/test/{0}.txt".format(i), 'w') as f:
    #                 js = json.loads(line)
    #                 temp = js['temp']
    #                 js['temp'] = change_length(temp, 100)
    #                 speed = js['speed']
    #                 js['speed'] = change_length(speed, 100)
    #                 power = js['power']
    #                 js['power'] = change_length(power, 100)
    #                 press = js['press']
    #                 js['press'] = change_length(press, 100)
    #                 f.write(json.dumps(js))
    #         print(i)
    # with open("my_dataset_change/train.txt", 'w') as f:
    #     for i in range(1, 160001):
    #         f.write("D:/wbw_d/menni/my_dataset_change/train/{0}.txt".format(i))
    #         f.write('\n')
    #         print(i)
    # with open("my_dataset_change/test.txt", 'w') as f:
    #     for i in range(160001, 200001):
    #         f.write("D:/wbw_d/menni/my_dataset_change/test/{0}.txt".format(i))
    #         f.write('\n')
    #         print(i)

    # with open("data/menni2107test.txt", 'r') as lines:
    #     i = 0
    #     for line in lines:
    #         i += 1
    #         with open("my_dataset_change/val/{0}.txt".format(i), 'w') as f:
    #             js = json.loads(line)
    #             temp = js['temp']
    #             js['temp'] = change_length(temp, 100)
    #             speed = js['speed']
    #             js['speed'] = change_length(speed, 100)
    #             power = js['power']
    #             js['power'] = change_length(power, 100)
    #             press = js['press']
    #             js['press'] = change_length(press, 100)
    #             f.write(json.dumps(js))
    #         print(i)
    #     length = i
    # with open("my_dataset_change/val.txt", 'w') as f:
    #     for i in range(1, length + 1):
    #         f.write("D:/wbw_d/menni/my_dataset_change/val/{0}.txt".format(i))
    #         f.write('\n')
    #         print(i)