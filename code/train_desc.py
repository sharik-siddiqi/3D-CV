from __future__ import print_function
import sys
sys.path.append("./Pointnet_Pointnet2_pytorch/models")

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
#from pointnet_sem_seg_desc import get_model_desc
#from pointnet_sem_seg import get_model
#from pointnet_utils import feature_transform_reguliarzer
from model import PointNetBasis as PointNetBasis
from model import PointNetDesc as PointNetDesc
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from dataload_light_rand import Surr12kModelNetDataLoader as DataLoader
import hdf5storage
from scipy.io import loadmat
import igl

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)
manualSeed = 1  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#_, f = igl.read_triangle_mesh("/home/raml_sharik/Diff-FMAPs-PyTorch-main/data/3793_simp_1000.obj")

b_size = 10

# Out Dir
outf = './models/trained/'
try:
    os.makedirs(outf)
except OSError:
    pass

a = np.arange(b_size)
b = list(np.arange(1,b_size))
b.append(0)

DATA_PATH = '/home/raml_sharik/Diff-FMAPs-PyTorch-main/data/'

TRAIN_DATASET = DataLoader(root=DATA_PATH, npoint=1000, split='train', uniform=True, normal_channel=False, augm = True)
TEST_DATASET = DataLoader(root=DATA_PATH, npoint=1000, split='test', uniform=True, normal_channel=False, augm = True)

dataset = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=b_size, shuffle=True, num_workers=0)
dataset_test = torch.utils.data.DataLoader(TEST_DATASET, batch_size=b_size, shuffle=True, num_workers=0)

basis = PointNetBasis(k=20, feature_transform=False)
#basis = get_model(20)
checkpoint = torch.load(outf + '/basis_model_best_mod_epoch_7.pth')
basis.load_state_dict(checkpoint)
basis.to(device)

classifier = PointNetDesc(k=40, feature_transform=False)
#classifier = get_model_desc(40)
checkpoint = torch.load(outf + '/desc_model_unsup_hk_0.01_0.007_epoch_33.pth')
classifier.load_state_dict(checkpoint)

optimizer = optim.Adam([{'params': classifier.parameters()}], lr=0.0001, betas=(0.9, 0.999))#
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

classifier.to(device)

best_eval_loss = np.inf;

criterion = nn.MSELoss(reduction='mean')

train_losses = [];
eval_losses = [];

iden_1 =  np.identity(20).reshape(1, 20, 20)
iden_1 = torch.Tensor(np.repeat(iden_1, b_size, axis=0))
iden_1 = iden_1.to(device)

# Descriptors loss
def desc_loss(phi_A, phi_B, G_A, G_B, area):
    p_inv_phi_A = torch.pinverse(phi_A)
    p_inv_phi_B = torch.pinverse(phi_B)

    c_G_A = torch.matmul(p_inv_phi_A, G_A)
    c_G_B = torch.matmul(p_inv_phi_B, G_B)

    #c_G_At = torch.transpose(c_G_A,2,1)
    #c_G_Bt = torch.transpose(c_G_B,2,1)
   
    C = torch.matmul(c_G_A, torch.pinverse(c_G_B))
    #C_2 = torch.matmul(c_G_B, torch.pinverse(c_G_A))                                               
    #C = torch.matmul(c_G_A,torch.transpose(torch.pinverse(c_G_Bt),2,1))       
    P = torch.matmul(phi_A, torch.matmul(C, torch.matmul(torch.transpose(phi_B, 2, 1), area)))  
    Q = F.normalize(P, 2, 1) ** 2
    
    return Q, C

# Training
for epoch in range(200):
    scheduler.step()
    train_loss = 0
    eval_loss = 0
    eucl_loss_list = [0, 0, 0, 0]
    eucl_loss_val = [0, 0, 0, 0]

    for data in tqdm(dataset, 0):
        points = data[0]
        dist = data[1]
        area_tensor = data[4]
        area_tensor = area_tensor.to(device)

        points = points.transpose(2, 1)
        points = points.to(device)
        dist = dist.to(device)
        optimizer.zero_grad()
        classifier = classifier.train()

        with torch.no_grad():
            basis = basis.eval()
            pred,_,_= basis(points)

        basis_A = pred[a,:,:]; basis_B = pred[b,:,:]
        dist_x = dist[a,:,:]; dist_y = dist[b,:,:]
        desc, trans, trans_feat =  classifier(points)
        desc_A = desc[a,:,:]; desc_B = desc[b,:,:];
        area_A = area_tensor[a,:,:]; area_B = area_tensor[b,:,:]

        Q, C = desc_loss(basis_A, basis_B, desc_A, desc_B, area_B)

        eucl_loss_1 = criterion(torch.bmm(Q.transpose(2,1), torch.bmm(dist_x, Q)) , dist_y)
        #eucl_loss_2 = criterion(torch.bmm(Q.transpose(2,1), Q) , iden_2)
        eucl_loss_4 = 0.1*criterion(torch.bmm(C.transpose(2,1), C), iden_1)
        #eucl_loss_5 = 0.1*criterion(torch.bmm(C_2.transpose(2,1), C_2), iden_1)
        #eucl_loss_6 = 0.1*criterion(torch.bmm(C_1, C_2), iden_1)
        #eucl_loss_7 = 0.1*criterion(torch.bmm(C_2, C_1), iden_1)
        #eucl_loss_8 = 0.01*feature_transform_reguliarzer(trans_feat)
        #eucl_loss_9 = 0.01*feature_transform_reguliarzer(trans)

        eucl_list = [eucl_loss_1.item(), eucl_loss_4.item()]
        eucl_loss = eucl_loss_1 + eucl_loss_4
  
        eucl_loss.backward()
        optimizer.step()
        train_loss += eucl_loss.item()
        for i in range(len(eucl_list)):
            eucl_loss_list[i] += eucl_list[i]
     
    print('Training Loss:', eucl_loss_list)

    for data in tqdm(dataset_test, 0):
        points = data[0]
        dist = data[1]
        dist = dist.to(device)
        points = points.transpose(2, 1)
        points = points.to(device)
        area_tensor = data[4]
        area_tensor = area_tensor.to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            basis = basis.eval()
            classifier = classifier.eval()
            pred,_,_ = basis(points)
            basis_A = pred[a,:,:]; basis_B = pred[b,:,:]
            desc, trans, trans_feat = classifier(points)
            desc_A = desc[a,:,:]; desc_B = desc[b,:,:]
            dist_x = dist[a, :, :]; dist_y = dist[b,:,:]
            area_A = area_tensor[a,:,:]; area_B = area_tensor[b,:,:]

            Q, C = desc_loss(basis_A, basis_B, desc_A, desc_B, area_B)
     
            eucl_loss_1 = criterion(torch.bmm(Q.transpose(2,1), torch.bmm(dist_x, Q)) , dist_y)
            #eucl_loss_2 = criterion(torch.bmm(Q.transpose(2,1), Q) , iden_2)
            eucl_loss_4 = 0.1*criterion(torch.bmm(C.transpose(2,1), C), iden_1)
            #eucl_loss_5 = 0.1*criterion(torch.bmm(C_2.transpose(2,1), C_2), iden_1)
            #eucl_loss_6 = 0.1*criterion(torch.bmm(C_1, C_2), iden_1)
            #eucl_loss_7 = 0.1*criterion(torch.bmm(C_2, C_1), iden_1)
            #eucl_loss_8 = 0.01*feature_transform_reguliarzer(trans_feat)
            #eucl_loss_9 = 0.01*feature_transform_reguliarzer(trans)
            eucl_val = [eucl_loss_1.item(), eucl_loss_4.item()]
            eucl_loss = eucl_loss_1 + eucl_loss_4
            eval_loss += eucl_loss.item()
            
            for i in range(len(eucl_val)):
                eucl_loss_val[i] += eucl_val[i]

    print('EPOCH ' + str(epoch) + ' - eva_loss: ' + str(eval_loss))
    print('Validation Loss:', eucl_loss_val)

    if eval_loss <  best_eval_loss:
        print('save model')
        best_eval_loss = eval_loss
        torch.save(classifier.state_dict(), '%s/desc_model_unsup_hk_0.01_0.007_0.005_epoch_{}.pth'.format(epoch) % (outf))
        #torch.save(basis.state_dict(), '%s/basis_model_best_20.pth' % (outf))

    train_losses.append(train_loss)
    eval_losses.append(eval_loss)

    np.save(outf+'/train_losses_desc_hk_0.01_0.007_0.005.npy',train_losses)
    np.save(outf+'/eval_losses_desc_hk_0.01_0.007_0.005.npy',eval_losses)
