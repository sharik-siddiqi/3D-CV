import numpy as np 
import warnings 
import os 
from torch.utils.data import Dataset 
from scipy.io import loadmat 
import random 
import math 
import igl 
#from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

def data_augmentation(point_set):
    theta = np.random.uniform(-np.pi/2, np.pi/2)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
    return point_set

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    return pc

def farthest_point_sample(point, npoint, count, match, rand):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    if(rand == True):
       count = count + 1
       N, D = point.shape
       xyz = point[:,:3]
       centroids = np.zeros((npoint + 400,))
       distance = np.ones((N,)) * 1e10
       farthest = np.random.randint(0, N)
       for i in range(npoint):
           centroids[i] = farthest
           centroid = xyz[farthest, :]
           dist = np.sum((xyz - centroid) ** 2, -1)
           mask = dist < distance
           distance[mask] = dist[mask]
           farthest = np.argmax(distance, -1) 
       centroids[:npoint] = match
       x = [i for i in range(xyz.shape[0]) if i not in centroids]
       centroids[npoint:npoint+400] = random.sample(x, k=400)
       point = point[centroids.astype(np.int32)]
       return point, centroids.astype(np.int32), count

    else:
       count = count + 1
       N, D = point.shape
       xyz = point[:,:3]
       centroids = np.zeros((npoint,))
       distance = np.ones((N,)) * 1e10
       farthest = np.random.randint(0, N)
       for i in range(npoint):
           centroids[i] = farthest
           centroid = xyz[farthest, :]
           dist = np.sum((xyz - centroid) ** 2, -1)
           mask = dist < distance
           distance[mask] = dist[mask]
           farthest = np.argmax(distance, -1)
       centroids[:npoint] = match
       #centroids[:npoint] = np.arange(2100)
       point = point[centroids.astype(np.int32),:]
       return point, centroids.astype(np.int32), count

class Surr12kModelNetDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000, augm = False, rand=False):
        self.uniform = uniform
        self.augm = augm
        self.indices = []
        self.count = 0
        self.npoints = npoint
        self.rand = rand
        self.split = split
         
        # Load the simplified mesh
        _, self.f = igl.read_triangle_mesh("/home/raml_sharik/Diff-FMAPs-PyTorch-main/data/3973_simp_1000.obj")
        
        # Matching co-ordinates from low density to high density meshes (for dealing with low density meshes (2100 -> 1000))
        match_fixed =  loadmat("/home/raml_sharik/Diff-FMAPs-PyTorch-main/data/match_2100_1000 (1).mat", variable_names = ['match'])
        match_fixed = match_fixed['match'][:,0] - 1
        self.match = match_fixed.reshape(-1,)
         
        # Matching co-ordinates from low density to high density meshes (for dealing with low density meshes (6890 -> 2100))
        match = loadmat("/home/raml_sharik/Diff-FMAPs-PyTorch-main/data/match_6890_2100 (1).mat", variable_names = ['match'])
        match = match['match'][:,0] - 1;
        
        num_test = 2 # Number of test data examples per sub-divided case (if 10 files, therefore total dataset size = 2*10 = 20)
        
        # A certain section of shapes taken from a bigger dataset ('vert_surreal_same.mat' & 'vert_bent_same.mat') (1000 shapes out of possible 4000 shapes)
        # {Work with the data you prepared as per the requirements given on the repository homepage readme.md}
        -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Divided datasets containing the indices (cases taken from a bigger dataset) and heat kernel data at t=0.005 sec and 2100 vertices/mesh
        # indices_0 = loadmat('./data/unsup_data_t_0.005_hk_0_same.mat', variable_names = ['indices'])
        # so on until indices_9 (10 files)
        # Overall dataset containg 4000 shapes (vertices and faces)
        # data_1 = loadmat(os.path.join(root, 'vert_surreal_same.mat'), variable_names = ['vert'])
        # data_2 = loadmat(os.path.join(root, 'vert_bent_same.mat'), variable_names = ['vert'])

        # Splitting into training and test dataset
        # if (split == 'train'):
        #    data_0_train = data_1['vert'][indices_0['indices'][0,:-num_test].reshape(-1,), :, :]
        #    data_bent_1_train = data_2['vert'][indices_8['indices'][0,:-num_test].reshape(-1,), :, :]

        # else:
        #    data_0_test = data_1['vert'][indices_0['indices'][0,-num_test:].reshape(-1,), :, :]
        #    data_bent_1_test = data_2['vert'][indices_8['indices'][0,-num_test:].reshape(-1,), :, :]
 

        # if split =="train":
        #    self.data -> concatenate all the files designated for training (data_0_train,  data_bent_1_train.....etc)

        # if split =="test":
        #    self.data -> concatenate all the files designated for validation (data_0_test,  data_bent_1_test.....etc)

        ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __len__(self):
        return len(self.data)

    def _get_item(self, index):
        point_set = self.data[index,:,:]
      
        if(self.split=='train'): 
          value = math.floor((index)/98)
          query = index%98
          
          if (value<8):
           g_dis = loadmat('/home/raml_sharik/Diff-FMAPs-PyTorch-main/data/unsup_data_t_0.005_hk_{}_same.mat'.format(value), variable_names = ['part_{}_{}'.format(value, query)])
           geo_set = g_dis['part_{}_{}'.format(value, query)]
          
          else:
           g_dis = loadmat('/home/raml_sharik/Diff-FMAPs-PyTorch-main/data/unsup_data_bent_t_0.005_hk_{}_same.mat'.format(value-8), variable_names = ['part_bent_{}'.format(query)])
           geo_set = g_dis['part_bent_{}'.format(query)]
          
          if self.uniform:
            point_set, self.indices, self.count = farthest_point_sample(point_set, self.npoints, self.count, self.match, self.rand)
            geo_set = geo_set[self.indices, :]
            geo_set = geo_set[:, self.indices]
            area = igl.massmatrix(point_set, self.f, igl.MASSMATRIX_TYPE_VORONOI).toarray()
            area = area/area.sum()
            #print(self.indices[:20])
            self.uniform = False
        
          else:
            point_set = point_set[self.indices,:]
            geo_set = geo_set[self.indices, :]
            geo_set = geo_set[:, self.indices]
            area = igl.massmatrix(point_set, self.f, igl.MASSMATRIX_TYPE_VORONOI).toarray()
            area = area/area.sum()
            self.count = self.count + 1
            #print(self.indices[:20])
            if(self.count%10==0):
               self.uniform = True

        else:
          value = math.floor((index)/2)
          query = 98 + (index%2)
          
          if (value<8):
           g_dis = loadmat('/home/raml_sharik/Diff-FMAPs-PyTorch-main/data/unsup_data_t_0.005_hk_{}_same.mat'.format(value), variable_names = ['part_{}_{}'.format(value, query)])
           geo_set = g_dis['part_{}_{}'.format(value, query)]
          
          else:
           g_dis = loadmat('/home/raml_sharik/Diff-FMAPs-PyTorch-main/data/unsup_data_bent_t_0.005_hk_{}_same.mat'.format(value-8), variable_names = ['part_bent_{}'.format(query)])
           geo_set = g_dis['part_bent_{}'.format(query)]
          
          if self.uniform:
            point_set, self.indices, self.count = farthest_point_sample(point_set, self.npoints, self.count, self.match, self.rand)
            geo_set = geo_set[self.indices, :]
            geo_set = geo_set[:, self.indices]
            area = igl.massmatrix(point_set, self.f, igl.MASSMATRIX_TYPE_VORONOI).toarray()
            area = area/area.sum()
            self.uniform = False
          
          else:
            point_set = point_set[self.indices]
            geo_set = geo_set[self.indices, :]
            geo_set = geo_set[:, self.indices]
            area = igl.massmatrix(point_set, self.f, igl.MASSMATRIX_TYPE_VORONOI).toarray()
            area = area/area.sum()
            self.count = self.count + 1
            if(self.count%10==0):
               self.uniform = True
        if self.augm:
            point_set = data_augmentation(point_set)
        point_set = pc_normalize(point_set)
        return  point_set.astype(np.float32), geo_set.astype(np.float32), index, self.indices.astype(np.int64), area.astype(np.float32)

    def __getitem__(self, index):
        return self._get_item(index)









