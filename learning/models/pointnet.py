"""
Code modified from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.nonlin = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.nonlin(self.bn1(self.conv1(x)))
        x = self.nonlin(self.bn2(self.conv2(x)))
        x = self.nonlin(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.nonlin(self.bn4(self.fc1(x)))
        x = self.nonlin(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=True, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)  # 64
        self.conv2 = torch.nn.Conv1d(64, 128, 1)   # 64, 128
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)  # 128, 1024
        self.bn1 = nn.BatchNorm1d(64)  # 64
        self.bn2 = nn.BatchNorm1d(128)  # 128
        self.bn3 = nn.BatchNorm1d(1024)  # 1024
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        
        self.nonlin = nn.ReLU()

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = self.nonlin(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = self.nonlin(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.mean(x, 2, keepdim=True)#[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetClassifier(nn.Module):
    def __init__(self, n_in):
        super(PointNetClassifier, self).__init__()

        self.feat = PointNetEncoder(global_feat=True, feature_transform=False, channel=n_in)
        self.fc1 = nn.Linear(1024, 512)  # 1024, 512
        self.fc2 = nn.Linear(512, 256)  # 512, 256
        self.fc3 = nn.Linear(256, 1)  # 256, 1
        self.dropout = nn.Dropout(p=0.)  # 0.4
        self.bn1 = nn.BatchNorm1d(512)  # 512
        self.bn2 = nn.BatchNorm1d(256)  # 256
        self.nonlin = nn.ReLU()

    def forward(self, x, zs):
        x, trans, trans_feat = self.feat(x)
        #x = torch.cat([x, zs], dim=1)
        x = self.nonlin(self.bn1(self.fc1(x)))
        #x = torch.cat([x, zs], dim=1)
        x = self.nonlin(self.bn2(self.dropout(self.fc2(x))))
        #x = torch.cat([x, zs], dim=1)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x #, trans_feat

class PointNetRegressor(nn.Module):
    def __init__(self, n_in, n_out):
        super(PointNetRegressor, self).__init__()

        self.feat = PointNetEncoder(global_feat=True, feature_transform=False, channel=n_in)
        self.fc1 = nn.Linear(1024, 512)  # 1024, 512
        self.fc2 = nn.Linear(512, 256)  # 512, 256
        self.fc3 = nn.Linear(256, n_out)  # 256, 1
        self.dropout = nn.Dropout(p=0.)  # 0.4
        self.bn1 = nn.BatchNorm1d(512)  # 512
        self.bn2 = nn.BatchNorm1d(256)  # 256
        self.nonlin = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.nonlin(self.bn1(self.fc1(x)))
        x = self.nonlin(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x

class PointNetPerPointClassifier(nn.Module):
    def __init__(self, n_in):
        super(PointNetPerPointClassifier, self).__init__()

        self.feat = PointNetEncoder(global_feat=False, feature_transform=False, channel=n_in)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.sigmoid(x)
        return x
