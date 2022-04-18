#Importing all the necessary libraries and functions 
from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from model import PointNetBasis as PointNetBasis                
from model import PointNetDesc as PointNetDesc
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from dataload_light_rand import Surr12kModelNetDataLoader as DataLoader
import hdf5storage
from scipy.io import loadmat
import igl

# Defining the GPU resource
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)

# Defining the randomness seed
manualSeed = 1  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Batch Size
b_size = 10

# Out Dir
outf = './models/trained/'
try:
    os.makedirs(outf)
except OSError:
    pass

# Pattern for comparison of shapes
a = np.arange(b_size)
b = list(np.arange(1,b_size))
b.append(0)

# Path where the dataset has been stored
DATA_PATH = '/home/raml_sharik/Diff-FMAPs-PyTorch-main/data/'

# Declaring instances of the custom datasets
TRAIN_DATASET = DataLoader(root=DATA_PATH, npoint=1000, split='train', uniform=True, normal_channel=False, augm = True)
TEST_DATASET = DataLoader(root=DATA_PATH, npoint=1000, split='test', uniform=True, normal_channel=False, augm = True)

# Using Torch DataLoader to define the batch_size and the shuffling parameters
dataset = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=b_size, shuffle=True, num_workers=0)
dataset_test = torch.utils.data.DataLoader(TEST_DATASET, batch_size=b_size, shuffle=True, num_workers=0)

# Using instance of a custom made deep learning model
basis = PointNetBasis(k=20, feature_transform=False)

# Loading predefined weights into the above model
checkpoint = torch.load(outf + '/basis_model_best_mod_select_epoch_63.pth')
basis.load_state_dict(checkpoint)

# network sent to the GPU resource
basis.to(device)

# Using instance of a custom made deep learning model
classifier = PointNetDesc(k=40, feature_transform=False)

# Defining the optimiser and the learning rate scheduler
optimizer = optim.Adam([{'params': classifier.parameters()}], lr=0.001, betas=(0.9, 0.999))#
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

#  network sent to the GPU resource
classifier.to(device)

best_eval_loss = np.inf;

# Instanstiating the Loss function
criterion = nn.MSELoss(reduction='mean')

train_losses = [];
eval_losses = [];

# Defining an identity matrix
iden_1 =  np.identity(20).reshape(1, 20, 20)
iden_1 = torch.Tensor(np.repeat(iden_1, b_size, axis=0))
iden_1 = iden_1.to(device)

# Descriptors Loss Function
def desc_loss(phi_A, phi_B, G_A, G_B, area):
    p_inv_phi_A = torch.pinverse(phi_A)
    p_inv_phi_B = torch.pinverse(phi_B)

    c_G_A = torch.matmul(p_inv_phi_A, G_A)
    c_G_B = torch.matmul(p_inv_phi_B, G_B)

    C = torch.matmul(c_G_A, torch.pinverse(c_G_B))
    P = torch.matmul(phi_A, torch.matmul(C, torch.matmul(torch.transpose(phi_B, 2, 1), area)))  
    Q = F.normalize(P, 2, 1) ** 2
    
    return Q, C

# Training
for epoch in range(800):

    # lr scheduler step
    scheduler.step()                                                 

    train_loss = 0
    eval_loss = 0
    eucl_loss_list = [0, 0]
    eucl_loss_val = [0, 0]

    # Batch-wise data
    for data in tqdm(dataset, 0):
        # 3D point data
        points = data[0]
     
        # Geodesic distance data                                
        dist = data[1]

        # Voronoi Area Data
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

        # Soft Correspondence Map Calculation 
        Q, C = desc_loss(basis_A, basis_B, desc_A, desc_B, area_B)

        # Final Loss Calculation
        eucl_loss_1 = criterion(torch.bmm(Q.transpose(2,1), torch.bmm(dist_x, Q)) , dist_y)
        eucl_loss_4 = 0.1*criterion(torch.bmm(C.transpose(2,1), C), iden_1)

        eucl_list = [eucl_loss_1.item(), eucl_loss_4.item()]
        eucl_loss = eucl_loss_1 + eucl_loss_4
  
        eucl_loss.backward()
        optimizer.step()

        train_loss += eucl_loss.item()
        for i in range(len(eucl_list)):
            eucl_loss_list[i] += eucl_list[i]
     
    print('Training Loss:', eucl_loss_list)

    # Validation
    # Batch-wise data
    for data in tqdm(dataset_test, 0):

        # 3D point data
        points = data[0]

        # Geodesic distance data
        dist = data[1]
        dist = dist.to(device)

        points = points.transpose(2, 1)
        points = points.to(device)

        # Voronoi Area Data
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

            # Soft Correspondence Map Calculation
            Q, C = desc_loss(basis_A, basis_B, desc_A, desc_B, area_B)
     
            # Final Loss Calculation
            eucl_loss_1 = criterion(torch.bmm(Q.transpose(2,1), torch.bmm(dist_x, Q)) , dist_y)
            eucl_loss_4 = 0.1*criterion(torch.bmm(C.transpose(2,1), C), iden_1)

            eucl_val = [eucl_loss_1.item(), eucl_loss_4.item()]
            eucl_loss = eucl_loss_1 + eucl_loss_4
            eval_loss += eucl_loss.item()
            
            for i in range(len(eucl_val)):
                eucl_loss_val[i] += eucl_val[i]

    print('EPOCH ' + str(epoch) + ' - eva_loss: ' + str(eval_loss))
    print('Validation Loss:', eucl_loss_val)

    # Storing better performing weights
    if eval_loss <  best_eval_loss:
        print('save model')
        best_eval_loss = eval_loss
        torch.save(classifier.state_dict(), '%s/desc_model_unsup_hk_0.01_epoch_{}.pth'.format(epoch) % (outf))

    train_losses.append(train_loss)
    eval_losses.append(eval_loss)

    #Storing Training/Validation Losses
    np.save(outf+'/train_losses_desc.npy',train_losses)
    np.save(outf+'/eval_losses_desc.npy',eval_losses)




