from __future__ import print_function 
import sys 
import argparse
import os
import random
import torch
import torch.optim as optim
from model import PointNetBasis
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from dataload import Surr12kModelNetDataLoader as DataLoader
from tqdm import tqdm
from random import sample
from PIL import Image
from scipy.io import loadmat
import torch.nn as nn
import igl
import torch.nn

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
manualSeed = 1  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Obtain the faces of one of the shapes in the dataset
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

# Load SHOT, HKS and WKS data for the training dataset shapes (Split into multiple files due to the large size) 
# {PREPARE ACCORDINGLY FOR YOUR OWN DATA}
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# shot_0 = loadmat('./shot_faust_0_2100_sam.mat') # Load necessary SHOT descriptor data for the dataset (if large dataset, divide into subsets and concatenate)
# hks_0 = loadmat('./hks_surreal_0_2100_sam.mat') # Load necessary HKS descriptor data for the dataset (if large dataset, divide into subsets and concatenate)
# wks_0 = loadmat('./pyshot/wks_surreal_0_2100_sam.mat') # Load necessary WKS descriptor data for the dataset (if large dataset, divide into subsets and concatenate)

# Train and test data split for the descriptor data
# shot_train -> concatenate all the subdivided SHOT descriptor files designated for training dataset (like shot_0 mentioned above)
# hks_train -> concatenate all the subdivided HKS descriptor files designated for training dataset (like hks_0 mentioned above)
# wks_train -> concatenate all the subdivided WKS descriptor files designated for training dataset (like wks_0 mentioned above)

# shot_test -> concatenate all the subdivided SHOT descriptor files designated for test dataset (like shot_0 mentioned above)
# hks_test -> concatenate all the subdivided SHOT descriptor files designated for test dataset (like hks_0 mentioned above)
# wks_test -> concatenate all the subdivided SHOT descriptor files designated for test dataset (like wks_0 mentioned above)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

desc_train = np.concatenate((shot_train,  wks_train), axis=2) # Take SHOT+WKS descriptors for our calculations
desc_test =  np.concatenate((shot_test,  wks_test), axis=2) # Take SHOT+WKS descriptors for our calculations

# Custom dataloader instance
TRAIN_DATASET = DataLoader(root=DATA_PATH, npoint=1000, split='train', uniform=True, normal_channel=False, augm = True, rand = False)
TEST_DATASET = DataLoader(root=DATA_PATH, npoint=1000, split='test', uniform=True, normal_channel=False, augm = True, rand = False)

dataset = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=b_size, shuffle=True, num_workers=0)
dataset_test = torch.utils.data.DataLoader(TEST_DATASET, batch_size=b_size, shuffle=True, num_workers=0)

# BasisNetwork with 20 basis
basisNet = PointNetBasis(k=20, feature_transform=False)

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
        eucl_loss_4 = 0.1*criterion(torch.bmm(C.transpose(2,1), C), iden_1)
        eucl_loss_5 = 0.1*criterion(torch.bmm(pred.transpose(2,1), torch.bmm(area_tensor, pred)), iden_1)
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
            eucl_loss_4 = 0.1*criterion(torch.bmm(C.transpose(2,1),C), iden_1)
            eucl_loss_5 = 0.1*criterion(torch.bmm(pred.transpose(2,1), torch.bmm(area_tensor, pred)), iden_1)
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

