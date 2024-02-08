import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy
import numpy as np
import random
import os


# from multi_class.config import load_config


def load_cifar100_superclass(is_train, shots=-1, superclass_type='predefined', target_superclass_idx=0,
                             n_classes=10, seed=0, reorganize=True):
    assert superclass_type in ('predefined', 'random', 'no_superclass')
    # config = load_config()
    # The mean and std could be different in different developers;
    # however, this will not influence the test accuracy much.
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    if is_train:
        transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # 暂无bug，需要传mean和std的参数
        transform = transforms.Compose([transforms.ToTensor(),
                                        normalize])

    dataset = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), train=is_train)

    if superclass_type == 'predefined':
        if 0 <= target_superclass_idx <= 19:
            sc2c = get_superclass2class_dict()
        elif 20 <= target_superclass_idx <= 39:
            sc2c = get_superclass2class_dict8()
        elif 40 <= target_superclass_idx <= 59:
            sc2c = get_superclass2class_dict4()
        elif 60 <= target_superclass_idx <= 79:
            sc2c = get_superclass2class_dict6()
        elif 80 <= target_superclass_idx <= 99:
            sc2c = get_superclass2class_dict7()
        else:
            sc2c = get_superclass2class_dict9(target_superclass_idx)
        dataset = _load_superclass_predefined(dataset, target_superclass_idx, sc2c, reorganize)
    elif superclass_type == 'random':
        dataset = _load_superclass_randomly(dataset, n_classes, seed, reorganize)
    elif superclass_type == 'no_superclass':  # Just for evaluating pretrained models on total test data.
        pass
    else:
        raise ValueError

    return dataset


def _load_superclass_predefined(dataset, target_superclass_idx, superclass2class, reorganize):
    classes = superclass2class[target_superclass_idx]
    dataset = extract_part_classes(dataset, classes, reorganize)
    return dataset


def _load_superclass_randomly(dataset, n_classes, seed, reorganize):
    random.seed(seed)
    classes = list(range(100))
    random.shuffle(classes)
    target_classes = classes[:n_classes]
    print(f'\nrandomly sampled classes: {target_classes}\n')

    dataset = extract_part_classes(dataset, target_classes, reorganize)
    return dataset


def get_superclass2class_dict():
    # Copy from https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    sc2c = dict()
    for c, sc in enumerate(coarse_labels):
        c_list = sc2c.get(sc, [])
        c_list.append(c)
        sc2c[sc] = c_list
    return sc2c


def get_superclass2class_dict9(target_superclass_idx):

    # Copy from https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
    if 100 <= target_superclass_idx <= 119:
        # four 0 + one 3:100
        coarse_labels = np.array([101, 118, 111, 105, 117, 103, 104, 107, 115, 100,
                                  103, 114, 106, 118, 107, 108, 103, 109, 107, 111,
                                  106, 111, 102, 110, 107, 106, 110, 112, 103, 115,
                                  100, 111, 101, 110, 109, 114, 113, 109, 111, 105,
                                  105, 116, 108, 108, 115, 113, 114, 114, 118, 107,
                                  116, 104, 117, 104, 119, 100, 117, 104, 118, 117,
                                  110, 103, 102, 112, 112, 116, 112, 101, 109, 119,
                                  102, 110, 100, 101, 116, 112, 109, 113, 115, 113,
                                  116, 119, 102, 104, 106, 119, 105, 105, 108, 119,
                                  118, 101, 102, 115, 106, 100, 117, 108, 114, 113])
    elif 120 <= target_superclass_idx <= 139:
        # three 0 + two 3:120
            coarse_labels = np.array([121, 138, 131, 125, 137, 123, 124, 124, 135, 120,
                                  120, 131, 126, 135, 127, 128, 123, 126, 127, 128,
                                  123, 131, 122, 127, 127, 126, 130, 132, 123, 132,
                                  137, 131, 138, 130, 129, 134, 133, 129, 131, 122,
                                  125, 136, 125, 128, 135, 130, 134, 134, 138, 127,
                                  133, 121, 134, 124, 139, 120, 137, 124, 138, 137,
                                  130, 123, 139, 129, 132, 136, 132, 121, 129, 139,
                                  122, 130, 120, 121, 136, 132, 129, 133, 135, 133,
                                  136, 136, 122, 124, 126, 139, 125, 125, 128, 139,
                                  138, 121, 122, 135, 126, 120, 137, 128, 134, 133])
    else:
        # four 0 + one 4:140
        coarse_labels = np.array([140, 157, 150, 144, 156, 142, 143, 147, 154, 159,
                                  143, 154, 145, 158, 147, 147, 143, 149, 147, 151,
                                  146, 151, 141, 146, 147, 146, 149, 151, 143, 155,
                                  140, 151, 141, 150, 148, 154, 152, 149, 151, 145,
                                  145, 155, 148, 148, 155, 153, 154, 153, 158, 150,
                                  156, 144, 157, 144, 158, 140, 157, 144, 158, 157,
                                  150, 143, 142, 152, 152, 156, 152, 141, 149, 159,
                                  142, 150, 140, 141, 156, 152, 149, 153, 155, 153,
                                  156, 159, 142, 144, 146, 159, 145, 145, 148, 159,
                                  158, 141, 142, 155, 146, 140, 157, 148, 154, 153])


    sc2c = dict()
    for c, sc in enumerate(coarse_labels):
        c_list = sc2c.get(sc, [])
        c_list.append(c)
        sc2c[sc] = c_list
    return sc2c






def get_superclass2class_dict8():
    #three 0 + two 2:20
    # Copy from https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
    coarse_labels = np.array([22, 39, 32, 26, 38, 24, 25, 25, 36, 21,
                              21, 32, 27, 36, 27, 29, 23, 27, 27, 29,
                              24, 31, 23, 28, 27, 26, 31, 33, 23, 33,
                              38, 31, 21, 28, 30, 34, 34, 29, 31, 23,
                              25, 37, 26, 28, 35, 31, 34, 35, 38, 30,
                              34, 22, 35, 24, 20, 20, 37, 24, 38, 37,
                              30, 23, 20, 30, 32, 36, 32, 21, 29, 37,
                              22, 30, 20, 21, 36, 32, 29, 33, 35, 33,
                              36, 39, 22, 24, 26, 39, 25, 25, 28, 39,
                              38, 39, 22, 35, 26, 20, 37, 28, 34, 33])
    sc2c = dict()
    for c, sc in enumerate(coarse_labels):
        c_list = sc2c.get(sc, [])
        c_list.append(c)
        sc2c[sc] = c_list
    return sc2c



def get_superclass2class_dict7():
    # Copy from https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
    coarse_labels = np.array([82, 81, 92, 86, 98, 84, 85, 87, 96, 81,
                              83, 94, 87, 98, 87, 89, 83, 89, 87, 91,
                              86, 91, 83, 88, 87, 86, 91, 93, 83, 95,
                              80, 91, 81, 90, 90, 94, 94, 89, 91, 85,
                              85, 97, 88, 88, 95, 93, 94, 95, 98, 90,
                              96, 84, 97, 84, 80, 80, 97, 84, 98, 97,
                              90, 83, 82, 92, 92, 96, 92, 81, 89, 99,
                              82, 90, 80, 81, 96, 92, 89, 93, 95, 93,
                              96, 99, 82, 84, 86, 99, 85, 85, 88, 99,
                              98, 99, 82, 95, 86, 80, 97, 88, 94, 93])
    sc2c = dict()
    for c, sc in enumerate(coarse_labels):
        c_list = sc2c.get(sc, [])
        c_list.append(c)
        sc2c[sc] = c_list
    return sc2c



def get_superclass2class_dict6():
    # Copy from https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
    coarse_labels = np.array([63, 60, 73, 67, 79, 65, 66, 66, 77, 62,
                              62, 73, 68, 77, 67, 70, 63, 68, 67, 70,
                              65, 71, 64, 69, 67, 66, 72, 74, 63, 74,
                              79, 71, 60, 69, 71, 74, 75, 69, 71, 64,
                              65, 78, 67, 68, 75, 72, 74, 76, 78, 70,
                              75, 63, 76, 64, 61, 60, 77, 64, 78, 77,
                              70, 63, 61, 71, 72, 76, 72, 61, 69, 78,
                              62, 70, 60, 61, 76, 72, 69, 73, 75, 73,
                              76, 79, 62, 64, 66, 79, 65, 65, 68, 79,
                              78, 61, 62, 75, 66, 60, 77, 68, 74, 73])
    sc2c = dict()
    for c, sc in enumerate(coarse_labels):
        c_list = sc2c.get(sc, [])
        c_list.append(c)
        sc2c[sc] = c_list
    return sc2c


def get_superclass2class_dict4():
    # Copy from https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
    coarse_labels = np.array([43, 40, 53, 47, 59, 45, 46, 47, 57, 42,
                              43, 54, 48, 58, 47, 50, 43, 49, 47, 51,
                              46, 51, 44, 49, 47, 46, 52, 54, 43, 55,
                              40, 51, 41, 50, 51, 54, 55, 49, 51, 45,
                              45, 58, 48, 48, 55, 53, 54, 56, 58, 50,
                              56, 44, 57, 44, 41, 40, 57, 44, 58, 57,
                              50, 43, 42, 52, 52, 56, 52, 41, 49, 59,
                              42, 50, 40, 41, 56, 52, 49, 53, 55, 53,
                              56, 59, 42, 44, 46, 59, 45, 45, 48, 59,
                              58, 41, 42, 55, 46, 40, 57, 48, 54, 53])
    sc2c = dict()
    for c, sc in enumerate(coarse_labels):
        c_list = sc2c.get(sc, [])
        c_list.append(c)
        sc2c[sc] = c_list
    return sc2c



def get_superclass2class_dict2():
    # Copy from https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
    coarse_labels = np.array([23, 20, 14, 27, 20, 25, 26, 26, 18, 22,
                              23, 14, 28, 18, 27, 29, 23, 28, 27, 30,
                              25, 31, 24, 29, 27, 26, 31, 15, 23, 15,
                              20, 31, 21, 30, 12, 14, 16, 29, 31, 25,
                              25, 19, 27, 28, 15, 13, 14, 17, 18, 30,
                              16, 24, 17, 24, 22, 20, 17, 24, 18, 17,
                              30, 23, 21, 31, 30, 16, 12, 21, 29, 19,
                              22, 10, 20, 21, 16, 12, 29, 13, 15, 13,
                              16, 19, 22, 24, 26, 19, 25, 5, 28, 19,
                              18, 21, 22, 15, 26, 0, 17, 28, 14, 13])
    sc2c = dict()
    for c, sc in enumerate(coarse_labels):
        c_list = sc2c.get(sc, [])
        c_list.append(c)
        sc2c[sc] = c_list
    return sc2c


def extract_part_classes(dataset, target_classes, reorganize):
    tc_data_list = []
    tc_targets_list = []
    for i, tc in enumerate(target_classes):
        tc_data_idx = np.where(np.array(dataset.targets) == tc)[0]
        tc_data = dataset.data[tc_data_idx]
        if reorganize:
            tc_targets = [i] * len(tc_data)
        else:
            tc_targets = [tc] * len(tc_data)
        tc_data_list.append(tc_data)
        tc_targets_list.append(tc_targets)
    tc_data = np.concatenate(tc_data_list, axis=0)
    tc_targets = np.concatenate(tc_targets_list, axis=0)

    dataset.data = tc_data
    dataset.targets = tc_targets

    idx2class = dict([(v, k) for k, v in dataset.class_to_idx.items()])
    dataset.classes = [idx2class[idx] for idx in target_classes]
    return dataset


if __name__ == '__main__':
    load_cifar100_superclass(is_train=True, superclass_type='random', n_classes=10, reorganize=True)
