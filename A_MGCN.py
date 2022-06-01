# Check in 2020-6-22(15:10)
import torch.nn as nn
import numpy as np
import torch


class A_MGCN(nn.Module):
    def __init__(self, n_input1=4096, n_input2=80, gcn_dim1=2048, gcn_dim2=32, n_bits=8,
                 batch_size=64, classes=80, belta=1):
        super(A_MGCN, self).__init__()
        self.batch_size = batch_size
        self.n_bits = n_bits
        self.belta = belta

        # GCN-feature
        self.conv1 = nn.Linear(n_input1, gcn_dim1)
        self.BN1 = nn.BatchNorm1d(gcn_dim1)
        self.act1 = nn.Tanh()

        # GCN-plabel
        self.conv2 = nn.Linear(n_input2, gcn_dim2)
        self.BN2 = nn.BatchNorm1d(gcn_dim2)
        self.act2 = nn.Tanh()

        # hashcode projection
        self.modal_1_hash = nn.Linear(gcn_dim1, n_bits)
        self.modal_2_hash = nn.Linear(gcn_dim2, n_bits)

        # Classify
        self.fc1 = nn.Linear(n_bits, classes)
        self.fc2 = nn.Linear(n_bits, classes)

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(n_bits, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_1, x_2, affinity):
        # GCN-1
        out_1 = self.conv1(x_1)
        out_1 = affinity.mm(out_1)
        out_1 = self.BN1(out_1)
        out_1 = self.act1(out_1)

        # GCN-2
        out_2 = self.conv2(x_2)
        out_2 = affinity.mm(out_2)
        out_2 = self.BN2(out_2)
        out_2 = self.act2(out_2)

        #  Get hash
        hash_1 = self.modal_1_hash(out_1)
        hash_2 = self.modal_2_hash(out_2)

        # pred for gcn output
        pred_1 = self.fc1(hash_1)
        pred_2 = self.fc2(hash_2)

        # concat hashcode
        hash = hash_1 + hash_2 * self.belta

        # discriminator
        D_1 = self.discriminator(hash_1)
        D_2 = self.discriminator(hash_2)

        return hash, D_1, D_2, pred_1, pred_2
