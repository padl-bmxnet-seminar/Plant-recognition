import os
import xml.etree.ElementTree as ET
import random
import argparse

label2num = {}


def read_data_dir(dir, root=""):
    src_list = []
    label_list = []
    for filename in os.listdir(os.path.join(root, dir)):
        if filename.endswith(".jpg"):
            src_list.append(os.path.join(os.path.join(root, dir), filename))
            continue
        else:
            if filename.endswith(".xml"):
                tree = ET.parse(os.path.join(os.path.join(root, dir), filename))
                tree_root = tree.getroot()
                label = tree_root.find('ClassId')
                label_list.append(map_label2num(label.text))
            continue
    return src_list, label_list


def map_label2num(label):
    if label not in label2num:
        label2num[label] = len(label2num)
    return label2num[label]


def write_data2file(src_list, label_list, prefix=""):
    with open(prefix + "_data_list.lst", 'w') as f:
        for i, (src, label) in enumerate(zip(src_list, label_list)):
            f.write(str(i) + "\t" + str(label) + "\t" + src + "\n")


def write_label2num(filename="label_map.lst"):
    with open(filename, 'w') as f:
        for key, val in label2num.items():
            f.write(key + ',' + str(val) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create train & val list for clef data set",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train_paths', nargs='+', default=["ImageCLEF2013PlantTaskTrainPackage-PART-1\\train",
                                                             "ImageCLEF2013PlantTaskTrainPackage-PART-2\\train"])
    parser.add_argument('--root_dir', default=os.path.abspath(os.path.dirname(__file__)))

    args = parser.parse_args()
    train_data_dirs = args.train_paths
    src_list = []
    label_list = []
    for dir in train_data_dirs:
        src, label = read_data_dir(dir, args.root_dir)
        src_list.extend(src)
        label_list.extend(label)
    shuffle_list = list(zip(src_list, label_list))
    random.shuffle(shuffle_list)
    src_list, label_list = zip(*shuffle_list)
    write_data2file(src_list[0:int(len(src_list) * .8)], label_list[0:int(len(label_list) * .8)], "train")
    write_data2file(src_list[int(len(src_list) * .8):], label_list[int(len(label_list) * .8):], "val")

    write_label2num()
