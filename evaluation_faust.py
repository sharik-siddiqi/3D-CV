import sys
#sys.path.append("./Pointnet_Pointnet2_pytorch/models")

#from pointnet2_sem_seg_msg import get_model
#from pointnet_utils import feature_transform_reguliarzer

from model import PointNetBasis
from model import PointNetDesc as PointNetDesc
import torch
import numpy as np
from scipy.io import savemat, loadmat
import random
from random import sample
from knnsearch import knnsearch
from our_match import our_match, our_match_desc
from comp_geo_error import calc_geo_err, comp_all_curve
import igl
from dataload_light_rand import pc_normalize, data_augmentation

device = torch.device("cpu")
DATA_PATH = '/home/raml_sharik/Diff-FMAPs-PyTorch-main/data/'

print(device)

# Loading Models
basis_model = PointNetBasis(k=20, feature_transform=False)
#basis_model = get_model(20)
desc_model = PointNetDesc(k=40, feature_transform=False)
#desc_model = get_model(40)

epoch = input('Which Epoch?')
epoch_basis = [63]
epoch_desc = [406]

v = loadmat('/home/raml_sharik/Diff-FMAPs-PyTorch-main/code/pyshot/faust_vertices_remeshed_2100.mat')
#f = loadmat('/home/raml_sharik/Diff-FMAPs-PyTorch-main/code/pyshot/faust_faces_remeshed_2100.mat')
geo_dist = loadmat('./pyshot/faust_geo_dist_2100_1.mat')
match =  loadmat('/home/raml_sharik/Diff-FMAPs-PyTorch-main/code/match_faust_2100_1000.mat')

_,f = igl.read_triangle_mesh('/home/raml_sharik/Diff-FMAPs-PyTorch-main/code/tr_reg_031_simplified_1000.ply')
#_,f = igl.read_triangle_mesh('/home/raml_sharik/Diff-FMAPs-PyTorch-main/code/tr_reg_031_simp_2100.ply')

#src_tar = loadmat('/home/raml_sharik/Diff-FMAPs-PyTorch/src_tar.mat')
#src_1 = src_tar['st'][0,:] - 1
#tar_1 = src_tar['st'][1,:] - 1

#src = np.concatenate((src_1, tar_1), axis=0)
#tar = np.concatenate((tar_1, src_1), axis=0)

geo_dist = geo_dist['geo']
v_clean = v['vertices_clean']
match = match['match_2100_1000'][:,0] - 1
order = np.argsort(v['indices'])

geo_dist = geo_dist[order[0,:],:,:]
v_clean = v_clean[order[0,:],:,:]

#match_not = []

#for i in range(2100):
#    if i not in match:
#      match_not.append(i)

#sam = random.sample(match_not, k=200)
#sam = np.array(sam).reshape(-1,)

#ind = np.concatenate((match, sam), axis=0)

v_clean = v_clean[:,match,:]
geo_dist = geo_dist[:,match,:]
geo_dist = geo_dist[:,:,match]

src = np.zeros((900,), dtype = np.int64)
tar = np.zeros((900,), dtype = np.int64)

for i in range(10):
  for j in range(1,10):
      s = list(range(i*10,(i+1)*10))
      t  = s[j:] + s[:j]
      src[(90*i + 10*(j-1)):(90*i + (j)*10)] = np.array(s)
      tar[(90*i + 10*(j-1)):(90*i + (j)*10)] = np.array(t)

for i in range(v_clean.shape[0]):
    #v_clean[i, :, :] = data_augmentation(v_clean[i,:,:])
    v_clean[i, :, :] = pc_normalize(v_clean[i,:,:])

mean_error_list = []

mean_error_start = np.inf

for idx_1,i in enumerate(epoch_basis):
    for idx_2,j in enumerate(epoch_desc):
      #epoch = input('Which Epoch?')
      checkpoint = torch.load('/home/raml_sharik/Diff-FMAPs-PyTorch-main/code/models/trained/basis_model_best_mod_select_epoch_{}.pth'.format(i))
      #checkpoint = torch.load('/home/raml_sharik/Diff-FMAPs-PyTorch-main/code/models/trained/basis_model_unsup_hk_0.01_epoch_{}.pth'.format(i))
      #checkpoint = torch.load('/home/raml_sharik/Diff-FMAPs-PyTorch-main/code/models/trained/basis_model_sup_high_sense_epoch_{}.pth'.format(i))
      basis_model.load_state_dict(checkpoint)
      #epoch = input('Which Epoch desc?')
      #checkpoint = torch.load('/home/raml_sharik/Diff-FMAPs-PyTorch-main/code/models/trained/desc_model_tnet_epoch_{}.pth'.format(j))
      #checkpoint = torch.load('/home/raml_sharik/Diff-FMAPs-PyTorch/code/models/trained/desc_model_best_epoch_{}.pth'.format(j))
      #checkpoint = torch.load('/home/raml_sharik/Diff-FMAPs-PyTorch-main/code/models/trained/desc_model_best_C_0.01_epoch_{}.pth'.format(j))
      #checkpoint = torch.load('/home/raml_sharik/Diff-FMAPs-PyTorch-main/code/models/trained/desc_model_best_pnpp_epoch_{}.pth'.format(j))
      checkpoint = torch.load('/home/raml_sharik/Diff-FMAPs-PyTorch-main/code/models/trained/desc_model_best_hk_0.01_epoch_{}.pth'.format(j))
      desc_model.load_state_dict(checkpoint)

      basis_model = basis_model.eval()
      desc_model = desc_model.eval()

# Computing Basis and Descriptors
      pred_basis = basis_model(torch.transpose(torch.from_numpy(v_clean.astype(np.float32)),1,2))
      pred_desc = desc_model(torch.transpose(torch.from_numpy(v_clean.astype(np.float32)),1,2))
      #pred_desc = desc_model(torch.transpose(pred_basis[0],1,2))
      #print(pred_basis[1]@pred_basis[1].transpose(2,1))
# Save Output
      dd_1 = pred_basis[0].detach().numpy()
      dd_2 = pred_desc[0].detach().numpy()

      match_opt = np.zeros((src.shape[0], v_clean.shape[1]), dtype = np.int32);
      match_desc_1 = np.zeros((src.shape[0], v_clean.shape[1]), dtype = np.int32);
      match_desc_2 = np.zeros((src.shape[0], v_clean.shape[1]), dtype = np.int32);
      geo_err_main = np.zeros((src.shape[0], v_clean.shape[1]), dtype = np.float32);
      match_phiM = np.zeros((src.shape[0], v_clean.shape[1], 20), dtype = np.float32);
      match_phiN = np.zeros((src.shape[0], v_clean.shape[1], 20), dtype = np.float32);

      for k in range(src.shape[0]):
          phiM = dd_1[src[k], :, :]
          phiN = dd_1[tar[k], :, :]
          match_opt[k,:] = our_match(phiM, phiN)
          descM = dd_2[src[k], :, :]
          descN = dd_2[tar[k], :, :]
          mat = our_match_desc(phiM, phiN, descM, descN)
          match_desc_1[k,:] = mat[0]
          match_phiM[k,:,:] = mat[1]
          match_phiN[k,:,:] = mat[2]
          match_desc_2[k,:] = mat[3]

      thr = np.linspace(0,1,1000)
      geo_err_main = np.zeros((src.shape[0], v_clean.shape[1]))

      for l in range(src.shape[0]):
          idx_src = src[l]
          idx_tar = tar[l]
          geo_dist_case = geo_dist[tar[l], :, :]
          errors = calc_geo_err([match_opt[l,:], match_desc_1[l,:], match_desc_2[l,:]], geo_dist_case)
          geo_err_main[l,:] = errors[:,1]
          if l==0:
             curves = comp_all_curve(errors, thr)
             mean_error = errors
          else:
             curves = curves + comp_all_curve(errors, thr)
             mean_error = mean_error + errors

      mean_curves = curves/src.shape[0];
      mean_error = np.mean(mean_error/src.shape[0], axis=0);

      mean_error_list.append([i, j, mean_error])

      #if(mean_error<mean_error_start):
      savemat('./curve_geo_error_non_iso.mat', {'match': match_desc_1, 'mean_curves' : mean_curves, 'thr' : thr, 'geo_err': geo_err_main, 'source':src, 'target':tar, 'vertices': v_clean, 'faces': f})
      #savemat('./curve_geo_error_basis.mat', {'vertices': v_clean, 'faces': f})     
      savemat('./dd2_non_iso.mat', {'desc':dd_2})
      #savemat('./match_phiM.mat', {'phiM': match_phiM})
      #savemat('./match_phiN.mat', {'phiN': match_phiN})

      #   mean_error_start = mean_error

      print(mean_error_list)
