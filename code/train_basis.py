from __future__ import print_function 
import sys 
sys.path.append("./Pointnet_Pointnet2_pytorch/models")

import argparse
import os
import random
import torch
import torch.optim as optim
from model import PointNetBasis
#from pointnet_sem_seg import get_model
#from pointnet_utils import feature_transform_reguliarzer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from dataload_light_rand import Surr12kModelNetDataLoader as DataLoader
from tqdm import tqdm
from random import sample
from PIL import Image
from scipy.io import loadmat
import torch.nn as nn
import igl
import torch.nn

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)
manualSeed = 1  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)

_, f = igl.read_triangle_mesh("/home/raml_sharik/Diff-FMAPs-PyTorch-main/data/3973_simp_2100.obj")

b_size = 10;

a = np.arange(b_size)
b = list(np.arange(1,b_size))
b.append(0)

# Out Dir
outf = './models/trained'
try:
    os.makedirs(outf)
except OSError:
    pass

DATA_PATH = '/home/raml_sharik/Diff-FMAPs-PyTorch-main/data/'

#num_test = 10

print("phase a")
shot_0 = loadmat('./pyshot/shot_faust_0_2100_sam.mat')
shot_1 = loadmat('./pyshot/shot_faust_1_2100_sam.mat')
shot_2 = loadmat('./pyshot/shot_faust_2_2100_sam.mat')
shot_3 = loadmat('./pyshot/shot_faust_3_2100_sam.mat')
shot_4 = loadmat('./pyshot/shot_faust_4_2100_sam.mat')
shot_5 = loadmat('./pyshot/shot_faust_5_2100_sam.mat')
shot_6 = loadmat('./pyshot/shot_faust_6_2100_sam.mat')
shot_7 = loadmat('./pyshot/shot_faust_7_2100_sam.mat')
shot_8 = loadmat('./pyshot/shot_faust_8_2100_sam.mat')
shot_9 = loadmat('./pyshot/shot_faust_9_2100_sam.mat')

print("phase b")
#hks_0 = loadmat('./pyshot/hks_surreal_0_2100_sam.mat')
#hks_1 = loadmat('./pyshot/hks_surreal_1_2100_sam.mat')
#hks_2 = loadmat('./pyshot/hks_surreal_2_2100_sam.mat')
#hks_3 = loadmat('./pyshot/hks_surreal_3_2100_sam.mat')
#hks_4 = loadmat('./pyshot/hks_surreal_4_2100_sam.mat')
#hks_5 = loadmat('./pyshot/hks_surreal_5_2100_sam.mat')
#hks_6 = loadmat('./pyshot/hks_surreal_6_2100_sam.mat')
#hks_7 = loadmat('./pyshot/hks_surreal_7_2100_sam.mat')
#hks_8 = loadmat('./pyshot/hks_surreal_8_2100_sam.mat')
#hks_9 = loadmat('./pyshot/hks_surreal_9_2100_sam.mat')

print("phase c")
wks_0 = loadmat('./pyshot/wks_surreal_0_2100_sam.mat')
wks_1 = loadmat('./pyshot/wks_surreal_1_2100_sam.mat')
wks_2 = loadmat('./pyshot/wks_surreal_2_2100_sam.mat')
wks_3 = loadmat('./pyshot/wks_surreal_3_2100_sam.mat')
wks_4 = loadmat('./pyshot/wks_surreal_4_2100_sam.mat')
wks_5 = loadmat('./pyshot/wks_surreal_5_2100_sam.mat')
wks_6 = loadmat('./pyshot/wks_surreal_6_2100_sam.mat')
wks_7 = loadmat('./pyshot/wks_surreal_7_2100_sam.mat')
wks_8 = loadmat('./pyshot/wks_surreal_8_2100_sam.mat')
wks_9 = loadmat('./pyshot/wks_surreal_9_2100_sam.mat')

shot_train = np.concatenate((shot_0['shot'][:-2,:,:], shot_1['shot'][:-2,:,:], shot_2['shot'][:-2,:,:], shot_3['shot'][:-2,:,:], shot_4['shot'][:-2,:,:], shot_5['shot'][:-2,:,:], shot_6['shot'][:-2,:,:], shot_7['shot'][:-2,:,:], shot_8['shot'][:-2,:,:], shot_9['shot'][:-2,:,:]), axis=0)
#shot_test = np.concatenate((shot_1['shot'][-num_test:,:,:], shot_2['shot'][-num_test:,:,:], shot_3['shot'][-num_test:,:,:], shot_4['shot'][-num_test:,:,:], shot_5['shot'][-num_test:,:,:], shot_6['shot'][-num_test:,:,:]), axis=0)
#hks_train = np.concatenate((hks_0['hks'][:-2,:,:], hks_1['hks'][:-2,:,:], hks_2['hks'][:-2,:,:], hks_3['hks'][:-2,:,:], hks_4['hks'][:-2,:,:], hks_5['hks'][:-2,:,:], hks_6['hks'][:-2,:,:], hks_7['hks'][:-2,:,:], hks_8['hks'][:-2,:,:], hks_9['hks'][:-2,:,:]), axis=0)
#hks_test = np.concatenate((hks_1['hks'][-num_test:,:,:], hks_2['hks'][-num_test:,:,:], hks_3['hks'][-num_test:,:,:], hks_4['hks'][-num_test:,:,:], hks_5['hks'][-num_test:,:,:], hks_6['hks'][-num_test:,:,:]), axis=0)
wks_train = np.concatenate((wks_0['wks'][:-2,:,:], wks_1['wks'][:-2,:,:], wks_2['wks'][:-2,:,:], wks_3['wks'][:-2,:,:], wks_4['wks'][:-2,:,:], wks_5['wks'][:-2,:,:], wks_6['wks'][:-2,:,:], wks_7['wks'][:-2,:,:], wks_8['wks'][:-2,:,:], wks_9['wks'][:-2,:,:]), axis=0)

shot_test = np.concatenate((shot_0['shot'][-2:,:,:], shot_1['shot'][-2:,:,:], shot_2['shot'][-2:,:,:], shot_3['shot'][-2:,:,:], shot_4['shot'][-2:,:,:], shot_5['shot'][-2:,:,:], shot_6['shot'][-2:,:,:], shot_7['shot'][-2:,:,:], shot_8['shot'][-2:,:,:], shot_9['shot'][-2:,:,:]), axis=0)
#hks_test = np.concatenate((hks_0['hks'][-2:,:,:], hks_1['hks'][-2:,:,:], hks_2['hks'][-2:,:,:], hks_3['hks'][-2:,:,:], hks_4['hks'][-2:,:,:], hks_5['hks'][-2:,:,:], hks_6['hks'][-2:,:,:], hks_7['hks'][-2:,:,:], hks_8['hks'][-2:,:,:], hks_9['hks'][-2:,:,:]), axis=0) 
wks_test = np.concatenate((wks_0['wks'][-2:,:,:], wks_1['wks'][-2:,:,:], wks_2['wks'][-2:,:,:], wks_3['wks'][-2:,:,:], wks_4['wks'][-2:,:,:], wks_5['wks'][-2:,:,:], wks_6['wks'][-2:,:,:], wks_7['wks'][-2:,:,:], wks_8['wks'][-2:,:,:], wks_9['wks'][-2:,:,:]), axis=0)

desc_train = np.concatenate((shot_train,  wks_train), axis=2)
desc_test =  np.concatenate((shot_test,  wks_test), axis=2)

#desc_train = shot_train;
#desc_test = shot_test;

TRAIN_DATASET = DataLoader(root=DATA_PATH, npoint=1000, split='train', uniform=True, normal_channel=False, augm = True, rand = False)
print("next one")
TEST_DATASET = DataLoader(root=DATA_PATH, npoint=1000, split='test', uniform=True, normal_channel=False, augm = True, rand = False)

dataset = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=b_size, shuffle=True, num_workers=0)
dataset_test = torch.utils.data.DataLoader(TEST_DATASET, batch_size=b_size, shuffle=True, num_workers=0)

# BasisNetwork with 20 basis
basisNet = PointNetBasis(k=20, feature_transform=False)
#basisNet = get_model(20)
#checkpoint = torch.load(outf + '/basis_model_unsup_hk_0.01_epoch_60.pth')
#basisNet.load_state_dict(checkpoint)

# Optimizer
optimizer = optim.Adam(basisNet.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
basisNet.to(device)

best_eval_loss = np.inf;

train_losses = [];
eval_losses = [];
faust_losses = [];

iden_1 =  np.identity(20).reshape(1, 20, 20)
iden_1 = torch.Tensor(np.repeat(iden_1, b_size, axis=0))
iden_1 = iden_1.to(device)

area = np.zeros((b_size, 1000, 1000))

# Descriptors loss
def desc_loss(phi_A, phi_B, G_A, G_B, area):
    p_inv_phi_A = torch.pinverse(phi_A)
    p_inv_phi_B = torch.pinverse(phi_B)

    c_G_A = torch.matmul(p_inv_phi_A, G_A)
    c_G_B = torch.matmul(p_inv_phi_B, G_B)

    #c_G_At = torch.transpose(c_G_A,2,1)
    #c_G_Bt = torch.transpose(c_G_B,2,1)

    C = torch.matmul(c_G_A, torch.pinverse(c_G_B))
    #C = torch.matmul(c_G_A,torch.transpose(torch.pinverse(c_G_Bt),2,1))
    P = torch.matmul(phi_A, torch.matmul(C, torch.matmul(torch.transpose(phi_B, 2, 1), area)))
    Q = F.normalize(P, 2, 1) ** 2

    return Q, C

# Training Loop
for epoch in range(800):
    scheduler.step()
    train_loss = 0
    eucl_loss_list = [0, 0, 0, 0, 0]
    eucl_loss_val = [0, 0, 0, 0, 0]
    criterion = nn.MSELoss(reduction='mean')

    # Training single Epoch
    for data in tqdm(dataset, 0):
        points = data[0]
        area_tensor = data[4]
        area_tensor = area_tensor.to(device)

        dist = data[1]
        desc = desc_train[data[2], :, :]
        desc = torch.Tensor(desc)
        points = points.transpose(2, 1)
        points = points.to(device)
        dist = dist.to(device)
        desc = desc.to(device)
        optimizer.zero_grad()
        basisNet = basisNet.train()

        # Obtaining predicted basis
        pred, trans, trans_feat = basisNet(points)

        # Generating pairs
        basis_A = pred[a,:,:]; basis_B = pred[b,:,:]
        dist_x = dist[a,:,:]; dist_y = dist[b,:,:]
        desc_A = desc[a,:,:]; desc_B = desc[b,:,:]
        desc_A = desc[:,data[3][0,:],:]; desc_B = desc[:,data[3][0,:],:]
        area_A = area_tensor[a,:,:]; area_B = area_tensor[b,:,:]

        #print(basis_A.shape, desc_A.shape)
        Q, C = desc_loss(basis_A, basis_B, desc_A, desc_B, area_B)
  
        eucl_loss_1 = criterion(torch.bmm(Q.transpose(2,1), torch.bmm(dist_x, Q)) , dist_y)
        #eucl_loss_3 = 0.1*criterion(torch.bmm(Q.transpose(2,1), Q), iden_2)
        eucl_loss_4 = 0.1*criterion(torch.bmm(C.transpose(2,1), C), iden_1)
        eucl_loss_5 = 0.1*criterion(torch.bmm(pred.transpose(2,1), torch.bmm(area_tensor, pred)), iden_1)
        #eucl_loss_6 = 0.1*criterion(torch.bmm(C_2.transpose(2,1), C_2), iden_1)
        #eucl_loss_7 = 0.1*criterion(torch.bmm(C_1, C_2), iden_1)
        #eucl_loss_8 = 0.1*criterion(torch.bmm(C_2, C_1), iden_1)
        #eucl_loss_9 = 0.01*feature_transform_reguliarzer(trans_feat)
        #eucl_loss_10 = 0.01*feature_transform_reguliarzer(trans)
        eucl_list = [eucl_loss_1.item(), eucl_loss_4.item(), eucl_loss_5.item()]
        eucl_loss = eucl_loss_1 + eucl_loss_4 + eucl_loss_5

        # Back Prop
        eucl_loss.backward()
        optimizer.step()
        train_loss += eucl_loss.item()

        for i in range(len(eucl_list)):
            eucl_loss_list[i] += eucl_list[i]

    print('Training Loss:', eucl_loss_list)

    # Validation
    with torch.no_grad():
        eval_loss = 0
        for data in tqdm(dataset_test, 0):
            points = data[0]
            area_tensor = data[4]
            area_tensor = area_tensor.to(device)
            
            dist = data[1]
            desc = desc_test[data[2],:,:]
            desc = torch.Tensor(desc)
            points = points.transpose(2, 1)
            points = points.to(device)
            dist = dist.to(device)
            desc = desc.to(device)
            basisNet = basisNet.eval()
            pred, trans, trans_feat = basisNet(points)
            
            basis_A = pred[a,:,:]; basis_B = pred[b,:,:]
            area_A = area_tensor[a,:,:]; area_B = area_tensor[b,:,:]            
            dist_x = dist[a,:,:]; dist_y = dist[b,:,:]
            desc_A = desc[a,:,:]; desc_B = desc[b,:,:]
            desc_A = desc_A[:,data[3][0,:],:];  desc_B = desc_B[:,data[3][0,:],:]
            
            Q, C = desc_loss(basis_A, basis_B, desc_A, desc_B, area_B)
            eucl_loss_1 = criterion(torch.bmm(Q.transpose(2,1), torch.bmm(dist_x, Q)) , dist_y)
            #eucl_loss_3 = criterion(torch.bmm(Q.transpose(2,1), Q), iden_2)
            eucl_loss_4 = 0.1*criterion(torch.bmm(C.transpose(2,1),C), iden_1)
            eucl_loss_5 = 0.1*criterion(torch.bmm(pred.transpose(2,1), torch.bmm(area_tensor, pred)), iden_1)
            #eucl_loss_6 = 0.1*criterion(torch.bmm(C_2.transpose(2,1), C_2), iden_1)
            #eucl_loss_7 = 0.1*criterion(torch.bmm(C_1, C_2), iden_1)
            #eucl_loss_8 = 0.1*criterion(torch.bmm(C_2, C_1), iden_1)
            #eucl_loss_9 = 0.01*feature_transform_reguliarzer(trans_feat)
            #eucl_loss_10 = 0.01*feature_transform_reguliarzer(trans)

            eucl_val = [eucl_loss_1.item(), eucl_loss_4.item(), eucl_loss_5.item()]
            eucl_loss = eucl_loss_1  + eucl_loss_4 + eucl_loss_5
            eval_loss +=  eucl_loss.item()
            
            for i in range(len(eucl_val)):
                eucl_loss_val[i] += eucl_val[i]

        print('EPOCH ' + str(epoch) + ' - eva_loss: ' + str(eval_loss))
        print('Validation Loss:', eucl_loss_val)

        # Saving if best model so far
        if eval_loss <  best_eval_loss:
            print('save model')
            best_eval_loss = eval_loss
            torch.save(basisNet.state_dict(), '%s/basis_model_unsup_hk_0.05_epoch_{}.pth'.format(epoch) % (outf))

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)

        # Logging losses
        #np.save(outf+'/train_losses_basis.npy',train_losses)
        #np.save(outf+'/eval_losses_basis.npy',eval_losses)

