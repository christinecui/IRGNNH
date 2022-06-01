from datasets import *
from A_MGCN import *
from utils import *
import metric_py

import torch
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
from scipy.io import savemat
import argparse

## param parse
parser = argparse.ArgumentParser()
parser.add_argument('--n_epoch', type=int, default=100, help='Epoch numbers.')
parser.add_argument('--nbit', type=int, default=48, help='Hash code length, also 16,32,48,64.')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Optimizer LearningRate')
parser.add_argument('--n_class', type=int, default=80, help='COCO/VOC/SUN are 80')
parser.add_argument('--dataset', type=str, default='VOC', help='COCO, NUSWIDE, VOC')
parser.add_argument('--alpha1', type=float, default=1, help='Supervise from Discrete B.')
parser.add_argument('--alpha2', type=float, default=0.0000001, help='D-1 loss.')
parser.add_argument('--alpha3', type=float, default=0.0000001, help='D-2 loss.')
parser.add_argument('--alpha4', type=float, default=0.1, help='GCN-1 classify loss.')
parser.add_argument('--alpha5', type=float, default=0.1, help='GCN-2 classify loss.')
parser.add_argument('--alpha7', type=float, default=0.00001, help='Similiar Loss.')
parser.add_argument('--belta', type=float, default=1, help='Hash-1 vs Hash-2.')
parser.add_argument('--times', type=int, default=1, help='Runtimes')

args = parser.parse_args()

## param
n_epoch = args.n_epoch
n_bits  = args.nbit
alpha_1 = args.alpha1
alpha_2 = args.alpha2
alpha_3 = args.alpha3
alpha_4 = args.alpha4
alpha_5 = args.alpha5
alpha_7 = args.alpha7
belta   = args.belta
n_class = args.n_class
dataset = args.dataset
print('Load %s data!'%(dataset))
print('Bits %d is running.'%(n_bits))
# (1) load
dset = 0
if dataset == 'COCO':
    dset = load_coco(n_bits, 'train')
elif dataset == 'VOC':
    dset = load_voc(n_bits, 'train')
else:
    raise Exception('No dataset can use!')

# (2) creat data
train_loader = data.DataLoader(my_dataset(dset.feature, dset.plabel),
    batch_size=256,
    shuffle=True)

# (3) model
gcn = A_MGCN(n_bits=n_bits, classes=n_class, belta=belta, n_input2=dset.plabel.shape[1])
gcn.cuda()

# (4) loss
loss_fn = torch.nn.MSELoss()
loss_cl = torch.nn.MultiLabelSoftMarginLoss()
loss_adv = torch.nn.BCELoss()

# (5) optimizer
optimizer = torch.optim.Adam(gcn.parameters(), lr=args.learning_rate)
# loss_data = []
import time
start_time = time.time() * 1000
for epoch in range(n_epoch):
    for i, (element, plabel) in enumerate(train_loader):
        _, aff = affinity_tag_multi(plabel.numpy(), plabel.numpy())
        lap_matrix = np.diag(np.sum(aff, axis=1)) - aff
        # CUDA
        aff = Variable(torch.Tensor(aff)).cuda()
        element = Variable(element).cuda()
        plabel = Variable(plabel).cuda()
        lap_matrix = Variable(torch.Tensor(lap_matrix)).cuda()

        # optimizer
        optimizer.zero_grad()
        hash, D1, D2, pred_1, pred_2 = gcn(element, plabel, aff)

        Binary = torch.sign(hash)
        ###  loss
        loss1 = loss_fn(hash, Binary) * alpha_1
        loss2 = loss_adv(D1, Variable(torch.zeros(D1.shape[0], 1)).cuda()) * alpha_2
        loss3 = loss_adv(D2, Variable(torch.ones(D2.shape[0], 1)).cuda()) * alpha_3
        loss4 = loss_cl(pred_1, plabel) * alpha_4
        loss5 = loss_cl(pred_2, plabel) * alpha_5
        loss7 = torch.trace(hash.t().mm(lap_matrix.mm(hash))) * alpha_7
        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss7

        loss.backward()
        optimizer.step()
        # loss_data.append(loss.item())

        if (i + 1) % 3 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Loss1: %.4f, Loss2: %.4f, Loss3: %.4f, Loss4: %.4f, Loss5: %.4f, Loss7: %.4f'
                  % (epoch + 1, n_epoch, i + 1, len(train_loader), loss.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss7.item()))

end_time = time.time() * 1000
print('Train times: ',(end_time-start_time) / 1000 )

# (5) get Hashcode
gcn.eval()
### Dont shuffle ### dont load train, val and retrieval at once!!!
################################### retrieval ##############################################
dset = 0
if dataset == 'COCO':
    dset = load_coco(n_bits, 'retrieval')
elif dataset == 'VOC':
    dset = load_voc(n_bits, 'retrieval')
else:
    raise Exception('No dataset can use!')

retrieval_loader = data.DataLoader(my_dataset(dset.feature, dset.plabel),
    batch_size=256,
    shuffle=False,
    num_workers=0)

retrievalP = []
for i, (element, plabel)  in enumerate(retrieval_loader):
    _, aff= affinity_tag_multi(plabel.numpy(), plabel.numpy())
    # CUDA
    aff = Variable(torch.Tensor(aff)).cuda()
    element = Variable(element).cuda()
    plabel = Variable(plabel).cuda()

    hash, D1, D2, pred_1, pred_2 = gcn(element, plabel, aff)
    retrievalP.append(hash.data.cpu().numpy())

retrievalH = np.concatenate(retrievalP)
retrievalCode = np.sign(retrievalH)
retrieval_label = dset.label

################################### val ##############################################
dset = 0
if dataset == 'COCO':
    dset = load_coco(n_bits, 'val')
elif dataset == 'VOC':
    dset = load_voc(n_bits, 'val')
else:
    raise Exception('No dataset can use!')

val_loader = data.DataLoader(my_dataset(dset.feature, dset.plabel),
    batch_size=256,
    shuffle=False,
    num_workers=0)

start_time = time.time()
valP = []
for i, (element, plabel) in enumerate(val_loader):
    _, aff = affinity_tag_multi(plabel.numpy(), plabel.numpy())
    # CUDA
    aff = Variable(torch.Tensor(aff)).cuda()
    element = Variable(element).cuda()
    plabel = Variable(plabel).cuda()

    hash, D1, D2, pred_1, pred_2 = gcn(element, plabel, aff)
    valP.append(hash.data.cpu().numpy())

valH = np.concatenate(valP)
valCode = np.sign(valH)
val_label = dset.label
end_time = time.time()
print('Test times: ', end_time-start_time)

# (6) curve and plot
HammTrainTest = 0.5 * (n_bits - np.dot(retrievalCode, valCode.T))
HammingRank = np.argsort(HammTrainTest, axis=0)
cateTrainTest = getTrue2(retrieval_label, val_label)

ap = metric_py.mAP(cateTrainTest, HammingRank, 1000)
pre, rec = metric_py.topK(cateTrainTest, HammingRank, 1000)

print('[%d bits] mAP: %.4f, precision@1000: %.4f, recall@1000: %.4f' % (n_bits, ap, pre, rec))
