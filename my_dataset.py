from __future__ import print_function
from matplotlib import lines
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import open3d as o3d

class My_Dataset(data.Dataset):
    """Wrapper class for loading data of a subset of ShapeNet Dataset"""
    def __init__(self,
                 path='./ShapeNetCore.v2/',
                 npoints=1024,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints # number of pointcloud sample points
        self.path = path # 数据集存储路径
        self.split = split
        self.data_augmentation = data_augmentation
        self.datapath = [] # 记录每一组数据的路径
        self.classes = [] # 记录所有标签
        
        print("\nStart create " + self.split + " dataset.")
        if split == 'train':
            self.labelfile = './train_list.txt' 
        elif split == 'test':
            self.labelfile = './test_list.txt' 
        else:
            print("Split Error!")
            return

        with open(self.labelfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                if self.npoints==5000:
                    pts_path = os.path.join(self.path,ls[0][15:-11],'models/model_normalized_.pts')
                else:
                    pts_path = os.path.join(self.path,ls[0][15:-11],'models/model_normalized_'+str(self.npoints)+'.pts')

                if not os.path.exists(pts_path):
                    pcd_path = os.path.join(self.path,ls[0][15:-11],'models/model_normalized.obj')
                    print(pcd_path)
                    mesh = o3d.io.read_triangle_mesh(pcd_path)
                    pcd = mesh.sample_points_uniformly(number_of_points=5000)
                    pcd = mesh.sample_points_poisson_disk(number_of_points=self.npoints, pcl=pcd)
                    o3d.io.write_point_cloud(pts_path,pointcloud=pcd,print_progress=True)
                    
                label = int(ls[1])
                if label not in self.classes:
                    self.classes.append(label)
                self.datapath.append((pts_path,label))
        print("Successfully created " + self.split + " dataset.")
        print("Total classes: " +str(len(self.classes)))
        print("Total data: " +str(len(self.datapath)) + "\n")
                
    def __getitem__(self, index):
        pts_path = self.datapath[index][0] # index对应数据的地址
        label = self.datapath[index][1]
        point_set = np.loadtxt(pts_path,skiprows=1).astype(np.float32)
        # print(point_set.shape)

        # resample
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist # scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter
            
        point_set = torch.from_numpy(point_set)
        cls = torch.from_numpy(np.array([label]).astype(np.int64))
        return point_set, cls
    
    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    # train_file = './train_list_rotate_5all_vol.txt'
    # train = []
    # with open(train_file,'r') as f1:
    #     for line in f1:
    #         if line[15:23] not in train:
    #             train.append(line[15:23])
    # print(train)
    # test_file = './test_list_rotate_5all_vol.txt'
    # test = []
    # with open(test_file,'r') as f2:
    #     for line in f2:
    #         if line[15:23] not in test:
    #             test.append(line[15:23])
    # print(test)
    
    Train_ShapeNet = My_Dataset(split='train',data_augmentation=True)
    Test_ShapeNet = My_Dataset(split='test',data_augmentation=False)
    pcd, label = Test_ShapeNet[0]
    print("PointCloud:")
    print(pcd.type)
    print(pcd.shape)
    print("Label: " + str(label))
    print(label.type)
    print(label.shape)
    
    # demo of sampling mesh to attain pointcloud
    # mesh = o3d.io.read_triangle_mesh('./ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.obj')
    # pcd = mesh.sample_points_uniformly(number_of_points=5000)
    # pcd = mesh.sample_points_poisson_disk(number_of_points=2500, pcl=pcd)
    # print(pcd.PointCloud)
    # o3d.io.write_point_cloud("./model.pts",pointcloud=pcd,print_progress=True)
    # point_set = np.loadtxt("./model.pts",skiprows=1).astype(np.float32)
    # point_set = torch.from_numpy(point_set)
    # print(point_set.shape)
    # print(point_set.dtype)
    
