from torch.utils.data import Dataset
from .utils import read_pfm 
import os
import glob
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T


# NOTE: INDEX in pairs.txt = IMAGE_ID - 1. image file names (based on the image ids start from 1 to...).
class DTUDataset(Dataset):
    def __init__(self, root_dir, split, n_views=3, levels=3, depth_interval=192.0, img_wh=None): # depth interval = depth range / number of hypothesis planes
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """
        self.root_dir = root_dir  # "C:\\Users\\Panagiotior\\Downloads\\eth3d"
        self.split = split
        assert self.split in ['train', 'val', 'test'], 'split must be either "train", "val" or "test"!'
        self.img_wh = img_wh
        if img_wh is not None:
            assert img_wh[0] % 32 == 0 and img_wh[1] % 32 == 0, 'img_wh must both be multiples of 32!'

        self.scans = ['facade']#, 'playground', 'courtyard', 'terrace']
        self.n_depths = depth_interval
        self.build_metas()
        self.n_views = n_views
        self.levels = levels  # FPN levels
        # self.depth_interval = {'facade': 1} #, 'playground': , 'courtyard': , 'terrace': }
        self.build_proj_mats()
        self.define_transforms()

        
    def build_metas(self):
        self.metas = []
        self.num_viewpoints_per_scan = {}

        for scan in self.scans:            
            pair_file = f"Cameras/{scan}/pair.txt"
            with open(os.path.join(self.root_dir, pair_file)) as f:                
                num_viewpoint = int(f.readline())
                self.num_viewpoints_per_scan[scan] = num_viewpoint  # save the number of viewpoints per scan in a dict
                for _ in range(num_viewpoint):  # facade images/viewpoints (76)
                    # NOTE: image id = index + 1
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]][:10] 
                    self.metas += [(scan, ref_view, src_views)] 

                    
    def build_proj_mats(self):
        self.proj_mats = {} # proj mats for each scan
        self.scale_factors = {}  # depth scale factors for each scan 

        for scan in self.scans:
            self.proj_mats[scan] = {}
            
            viewpoints = self.num_viewpoints_per_scan[scan]   # for facade images/viewpoints (76)
            for vid in range(0, viewpoints): 
                proj_mat_filename = glob.glob(os.path.join(self.root_dir, f'Cameras/{scan}/train/DSC_*_id{vid+1:04d}_cam.txt'))[0] # returns a list
                intrinsics, extrinsics, depth_min = self.read_cam_file(scan, proj_mat_filename)

                # resize the intrinsics to the coarsest level
                intrinsics[0] *= 1440 / 4  # width (x)
                intrinsics[1] *= 960 / 4   # height (y)

                # same viewpoint projection matrix for 3-levels
                proj_mat_ls = []
                for l in reversed(range(self.levels)):  # 2-->1-->0
                    proj_mat_l = np.eye(4)
                    proj_mat_l[:3, :4] = intrinsics @ extrinsics[:3, :4]
                    intrinsics[:2] *= 2  # 1/4->1/2->1
                    proj_mat_ls += [torch.FloatTensor(proj_mat_l)] # appends new tensor to list. [tensor_l2, tensor_l1, tensor_l0]
                # (self.levels, 4, 4) from fine to coarse 0-->1-->2
                proj_mat_ls = torch.stack(proj_mat_ls[::-1]) # torch.Size([3, 4, 4]) 
                self.proj_mats[scan][vid] = (proj_mat_ls, depth_min)    # double dictionary          


    def read_cam_file(self, scan, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        if scan not in self.scale_factors:
            # use the first cam to determine scale factor for each scan.
            self.scale_factors[scan] = 100/depth_min  # scale * depth_min = 100
        
        # unique depth min, common scaling factor 
        depth_min *= self.scale_factors[scan]
        extrinsics[:3, 3] *= self.scale_factors[scan] # scales the translation vector too (col 4th).
        return intrinsics, extrinsics, depth_min


    def read_depth(self, scan, filename):
        depth_0 = np.array(read_pfm(filename)[0], dtype=np.float32)  # (960, 1440) = (h, w)
        
        # scales the depths (distances)
        depth_0 *= self.scale_factors[scan]
        depth_1 = cv2.resize(depth_0, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        depth_2 = cv2.resize(depth_1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

        depths = {"level_0": torch.FloatTensor(depth_0),
                  "level_1": torch.FloatTensor(depth_1),
                  "level_2": torch.FloatTensor(depth_2)}

        # finds maximum depth (distance) of the specific viewpoint
        depth_max = depth_0.max()
        return depths, depth_max

    
    def read_mask(self, filename):
        mask_0 = cv2.imread(filename, 0)  # (960, 1440). cv2 returns (height, width).
        mask_1 = cv2.resize(mask_0, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)  # (480, 720).
        mask_2 = cv2.resize(mask_1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)  # (240, 360).

        masks = {"level_0": torch.BoolTensor(mask_0),
                 "level_1": torch.BoolTensor(mask_1),
                 "level_2": torch.BoolTensor(mask_2)}
        return masks


    def define_transforms(self):
            # you can add augmentation here
            self.transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.metas)


    def __getitem__(self, idx):
        sample = {}
        scan, ref_view, src_views = self.metas[idx] 
        view_ids = [ref_view] + src_views[:self.n_views - 1]  # [ref view, src1, src2] indices

        imgs = []       # 3 images: [ref_view, src1, src2]
        proj_mats = []  # homography matrices (record proj mats between views)
        
        for i, vid in enumerate(view_ids): # repeats for all 3 views 

            # the id in image file names is from 1 to ... (not 0~...). Use vid+1. 
            img_filename = glob.glob(os.path.join(self.root_dir, f'Rectified/{scan}/rect_DSC_*_id{vid+1:04d}.png'))[0]
            depth_filename = glob.glob(os.path.join(self.root_dir, f'Depths/{scan}/interp_depth_map_DSC_*_id{vid+1:04d}.pfm'))[0]
            mask_filename = glob.glob(os.path.join(self.root_dir, f'Depths/{scan}/seg_depth_visual_DSC_*_id{vid+1:04d}.png'))[0]
            
            img = Image.open(img_filename)  
            img = self.transform(img)
            imgs += [img]   # 3 images: [ref_view, src1, src2]

            # collects the projection matrices of all 3 views (ref, src1, src2) by the end of the loop 
            proj_mat_ls, depth_min = self.proj_mats[scan][vid]
            
            if i == 0:  # reference view            
                # Collects only the mask image, depth map, and depth_min of the reference viewpoint.

                sample['masks'] = self.read_mask(mask_filename)
                sample['depths'], depth_max = self.read_depth(scan, depth_filename)

                # Define the depth interval of the MVS.                
                depth_interval = (depth_max - depth_min) / self.n_depths
                sample['init_depth_min'] = torch.FloatTensor([depth_min])
                sample['depth_interval'] = torch.FloatTensor([depth_interval])  
                print("depth interval: ", depth_interval)

                ref_proj_inv = torch.inverse(proj_mat_ls)
            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]  # contains two elements.

        imgs = torch.stack(imgs)  # (V=3, 3, H, W)
        proj_mats = torch.stack(proj_mats)[:, :, :3]  # (V-1, self.levels, 3, 4) from fine to coarse
        # V-1 CAUSE IT CONTAINS ONLY THE TWO FINAL PROJ MATRICES OF THE TWO SOURCE IMAGES. BASED ON THE REFERENCE PROJ MAT.

        sample['imgs'] = imgs # (V, 3, H, W)
        sample['proj_mats'] = proj_mats  # (V-1, self.levels, 3, 4)
        sample['scan_vid'] = (scan, ref_view) # vid stands for viewid.

        return sample


dtu_dataset = DTUDataset(root_dir = "/scratch/ipanagiotidou/data/mvs_training/eth3d", split = 'train', n_views=3, levels=3, img_wh=None) # depth_interval=200.0
#print(dtu_dataset[11])
