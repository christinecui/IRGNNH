import torch.utils.data as data
from torch.autograd import Variable
import torch
import pickle as pkl
import numpy as np
from collections import namedtuple
from itertools import permutations
from sklearn import preprocessing

from utils import *
from scipy.io import loadmat

## COCO-data
paths_COCO = {
    'COCO_train': './pkl/COCO/train2017.pkl',
    'COCO_val': './pkl/COCO/val2017.pkl',
    'COCO_retrieval': './pkl/COCO/retrieval2017.pkl',
}
## VOC-data
paths_VOC = {
    'VOC_train': './pkl/VOC/train2012.pkl',
    'VOC_val': './pkl/VOC/val2012.pkl',
    'VOC_retrieval': './pkl/VOC/retrieval2012.pkl',
}

dataset_lite = namedtuple('dataset_lite', ['feature', 'label', 'plabel'])


def load_coco(n_bits, mode):
    if mode == 'train':
        data = pkl.load(open(paths_COCO['COCO_train'], 'rb'))
        feature = data['train_feature']
        label = data['train_label']
        plabel = data['train_tag']

    elif mode == 'retrieval':
        data = pkl.load(open(paths_COCO['COCO_retrieval'], 'rb'))
        feature = data['retrieval_feature']
        label = data['retrieval_label']
        plabel = data['retrieval_tag']

    else:
        data = pkl.load(open(paths_COCO['COCO_val'], 'rb'))
        feature = data['val_feature']
        label = data['val_label']
        plabel = data['val_tag']

    return dataset_lite(feature, label, plabel)

def load_voc(n_bits, mode):
    if mode == 'train':
        data = pkl.load(open(paths_VOC['VOC_train'], 'rb'))
        feature = data['train_feature']
        label = data['train_label']
        plabel = data['train_tag']

    elif mode == 'retrieval':
        data = pkl.load(open(paths_VOC['VOC_retrieval'], 'rb'))
        feature = data['retrieval_feature']
        label = data['retrieval_label']
        plabel = data['retrieval_tag']

    else:
        data = pkl.load(open(paths_VOC['VOC_val'], 'rb'))
        feature = data['val_feature']
        label = data['val_label']
        plabel = data['val_tag']

    return dataset_lite(feature, label, plabel)

class my_dataset(data.Dataset):
    def __init__(self, feature, plabel):
        self.feature = torch.Tensor(feature)
        self.plabel = torch.Tensor(plabel)
        self.length = self.feature.size(0)

    def __getitem__(self, item):
        return self.feature[item, :], self.plabel[item, :]

    def __len__(self):
        return self.length

