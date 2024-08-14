"""
Code modified from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable


# write a dummy no transform module

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)  # 64
        self.conv2 = torch.nn.Conv1d(64, 128, 1)  # 64, 128
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)  # 128, 1024
        self.fc1 = nn.Linear(1024, 512)  # 1024, 512
        self.fc2 = nn.Linear(512, 256)  # 512, 256
        self.fc3 = nn.Linear(256, 9)  # 256, 9
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
        x = iden + x
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
        x = torch.linalg.qr(x)[0]  # Make sure X is a valid rotation matrix.
        return x

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=True, num_geom_features=1,
                 channel=3, n_out=1024, use_batch_norm=False, use_stn=True):
        super(PointNetEncoder, self).__init__()

        self.num_geom_feats = num_geom_features

        self.use_stn = use_stn
        if use_stn:
            self.stn = STN3d(channel)

        self.n_out = n_out
        n_hidden1, n_hidden2 = n_out // 16, n_out // 8

        self.conv1 = torch.nn.Conv1d(channel, n_hidden1, 1)  # 64
        self.conv2 = torch.nn.Conv1d(n_hidden1, n_hidden2, 1)  # 64, 128
        self.conv3 = torch.nn.Conv1d(n_hidden2, n_out, 1)  # 128, 1024
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(n_hidden1)  # 64
            self.bn2 = nn.BatchNorm1d(n_hidden2)  # 128
            self.bn3 = nn.BatchNorm1d(n_out)  # 1024
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=n_hidden1)

        self.nonlin = nn.ReLU()

    def forward(self, x, override_transform=None):
        B, D, N = x.size()

        # if we are co-opting a transform that has been produce in some other part of the process,
        # then use the one passed instead
        if override_transform is None:
            if self.use_stn: 
                trans = self.stn(x)
            else: 
                trans = None 
        else:
            trans = override_transform

        x = x.transpose(2, 1)
        if trans is not None:
            if D > 3:
                non_geometric_feats = x[:, :, self.num_geom_feats*3:]
                geom_feats = []
                for i in range(self.num_geom_feats):
                    geom_feats.append(torch.bmm(x[:, :, i * 3:(i + 1) * 3], trans))
                x = torch.cat([*geom_feats, non_geometric_feats], dim=2)
            else:
                x = torch.bmm(x, trans)

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
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.n_out)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.n_out, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetClassifier(nn.Module):
    def __init__(self, n_in, n_geometric_features=1, use_stn=True):
        super(PointNetClassifier, self).__init__()

        n_enc = 1024
        self.feat = PointNetEncoder(global_feat=False, feature_transform=False, channel=n_in, n_out=n_enc,
                                    use_batch_norm=False, num_geom_features=n_geometric_features, use_stn=use_stn)

        self.fc1 = nn.Linear(n_enc + n_enc // 16, n_enc // 2)  # 1024, 512
        self.fc2 = nn.Linear(n_enc // 2, n_enc // 4)  # 512, 256
        self.fc3 = nn.Linear(n_enc // 4, 1)  # 256, 1
        self.dropout = nn.Dropout(p=0.)  # 0.4
        self.bn1 = nn.Identity()  # nn.BatchNorm1d(n_enc//2)  # 512
        self.bn2 = nn.Identity()  # nn.BatchNorm1d(n_enc//4)  # 256
        self.nonlin = nn.ReLU()

    def forward(self, x, zs, override_transform=None):
        x, trans, trans_feat = self.feat(x, override_transform=override_transform)
        n_batch, n_feat, n_pts = x.shape
        x = x.swapaxes(1, 2).reshape(-1, n_feat)
        x = self.nonlin(self.bn1(self.fc1(x)))
        x = self.nonlin(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = x.view(n_batch, n_pts, x.shape[-1])
        x = x.mean(dim=1)
        x = torch.sigmoid(x)
        return x, trans

    # TODO: add option to add a transform override


class PointNetRegressor(nn.Module):
    def __init__(self, n_in, n_out, n_geometric_features=1, use_batch_norm=False, use_stn=True):
        super(PointNetRegressor, self).__init__()

        n_enc = 1024
        self.feat = PointNetEncoder(
            global_feat=False,
            feature_transform=False,
            channel=n_in,
            n_out=n_enc,
            use_batch_norm=use_batch_norm,
            num_geom_features=n_geometric_features,
            use_stn=use_stn
        )

        self.fc1 = nn.Linear(n_enc + n_enc // 16, n_enc // 2)  # 1024, 512
        self.fc2 = nn.Linear(n_enc // 2, n_enc // 4)  # 512, 256
        self.fc3 = nn.Linear(n_enc // 4, n_out)  # 256, 1
        self.dropout = nn.Dropout(p=0.2)  # 0.4
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(n_enc // 2)  # 512
            self.bn2 = nn.BatchNorm1d(n_enc // 4)  # 256
        else:
            self.bn1 = nn.Identity()  # nn.BatchNorm1d(n_enc//2)  # 512
            self.bn2 = nn.Identity()  # nn.BatchNorm1d(n_enc//4)  # 256
        self.nonlin = nn.ReLU()

    def forward(self, x, override_transform=None):
        x, trans, trans_feat = self.feat(x, override_transform=override_transform)
        n_batch, n_feat, n_pts = x.shape
        x = x.swapaxes(1, 2).reshape(-1, n_feat)
        x = self.nonlin(self.bn1(self.fc1(x)))
        x = self.nonlin(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = x.view(n_batch, n_pts, x.shape[-1])
        x = x.mean(dim=1)  # TODO: this may cause input number to be lost, may need to substitute for an operation that
        return x, trans
