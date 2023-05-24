from torch.utils.data import Dataset
from .utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T

class DTUDataset(Dataset):
    def __init__(self, root_dir, split, n_views=3, levels=3, depth_interval=2.65,
                 img_wh=None):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'val', 'test'], \
            'split must be either "train", "val" or "test"!'
        self.img_wh = img_wh
        if img_wh is not None:
            assert img_wh[0]%32==0 and img_wh[1]%32==0, \
                'img_wh must both be multiples of 32!'
        self.build_metas()
        self.n_views = n_views
        self.levels = levels # FPN levels
        self.depth_interval = depth_interval
        self.build_proj_mats()
        self.define_transforms()

    def build_metas(self):
        self.metas = []
        with open(f'datasets/lists/dtu/{self.split}.txt') as f:
            self.scans = [line.rstrip() for line in f.readlines()]

        # light conditions 0-6 for training
        # light condition 3 for testing (the brightest?)
        light_idxs = [3] if self.img_wh else range(7)

        pair_file = "Cameras/pair.txt"
        for scan in self.scans:
            with open(os.path.join(self.root_dir, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    for light_idx in light_idxs:
                        self.metas += [(scan, light_idx, ref_view, src_views)]

    def build_proj_mats(self):
        proj_mats = []
        for vid in range(49): # total 49 view ids
            if self.img_wh is None: # I: when we train it is None 
                proj_mat_filename = os.path.join(self.root_dir,
                                                 f'Cameras/train/{vid:08d}_cam.txt')
            else:
                proj_mat_filename = os.path.join(self.root_dir,
                                                 f'Cameras/{vid:08d}_cam.txt')
            intrinsics, extrinsics, depth_min = \
                self.read_cam_file(proj_mat_filename)
            # I: if img_wh is not None, it means we are in test mode 
            if self.img_wh is not None: # resize the intrinsics to the coarsest level  
                # I: intrinsic camera parameters are needed for calculating the projection of 3D point on to the image plane
                # I: This operation adjusts the intrinsic matrix by scaling its values based on the downscaling factors. It ensures that the intrinsics remain consistent with the current image size. 
                intrinsics[0] *= self.img_wh[0]/1600/4       # I: dividing the img_width (current image width)/ 1600 (original image width) calculates the scaling factor by which the current image differs from the refence image (0.5 for example). 
                intrinsics[1] *= self.img_wh[1]/1200/4       # I: dividing again with a scaling factor of 4 it suggests that the current image has been downscaled 4 times compared to the reference width. 
                # Ι: έχω για παράδειγμα ένα input size για την test image = 1200. Τότε 1200/1600 = 0.75 scaling factor και αν επιπλέον αυτή την image την κάνω downscale by 4 (στα 300 pixels) τότε θα έχω 0.75/4 = 0.1875 
                # που είναι ίσο με 300/1600 = 0.1875.
                
                # I: Intrinsic camera parameters for resized images: * Cropping FROM THE CENTER has No Effect (The principle point is still at the center).
                # I: Rescale the intrinsic matrix. For example, if the original camera image is 1280 x 960 and resized image is 320 x 240, the ratio is 1/4 and the focal length and principal point is scaled so.
                # Αν απομακρυνθεί το image plane από το principal point 2 φορές (scaling factor = 2), τότε τόσο το focal length μεγαλώνει κατά 2 και οι συντεταγμένες του σημείου απομακρύνονται από την αρχή των αξόνων του image plane.   
                
                # Example:
                # Let's say you have resized image (cv2.resize) and changed aspect ratio. e.g. was 1242 x 375 now 512 x 256. 
                # Then you need to multiply first row (x) with width scale scale_x = 512 / 1242 and second row with height scale scale_y = 256 / 375
                # K[0]*=scale_x
                # K[1]*=scale_y
                

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_ls = []
            for l in reversed(range(self.levels)): # values go from {2 --> 1 --> 0} 
                proj_mat_l = np.eye(4)
                proj_mat_l[:3, :4] = intrinsics @ extrinsics[:3, :4]
                intrinsics[:2] *= 2 # 1/4->1/2->1   # I: we upscale (double) the focal length and shift the principal point to match the upscaled image size. 
                proj_mat_ls += [torch.FloatTensor(proj_mat_l)]
            # (self.levels, 4, 4) from fine to coarse
            proj_mat_ls = torch.stack(proj_mat_ls[::-1])  # I: stacks from {0--> 1 --> 2} (from fine to coarse)
            proj_mats += [(proj_mat_ls, depth_min)]

        self.proj_mats = proj_mats

    def read_cam_file(self, filename):
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
        return intrinsics, extrinsics, depth_min

    def read_depth(self, filename):
        # I: from MVSNet: "we downsize the image resolution from 1600x1200 to 800x600 and then CROP the image patch with W=640 and H=512 FROM THE CENTER. The input CAMERA PARAMS are changed accordingly." 
        depth = np.array(read_pfm(filename)[0], dtype=np.float32) # (1200, 1600)
        if self.img_wh is None: # I: when we train it is None 
            depth = cv2.resize(depth, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST) # (600, 800)   # I: with resizing the number of pixels gets smaller (half here, from 1200 to 600 in one dimension). The image becomes smaller but the FOV remains the same. 
            depth_0 = depth[44:556, 80:720] # (512, 640)                    # I: we crop it. The image resolution is set to (640 x 512).
        else:
            depth_0 = cv2.resize(depth, self.img_wh,
                                 interpolation=cv2.INTER_NEAREST)
        depth_1 = cv2.resize(depth_0, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)
        depth_2 = cv2.resize(depth_1, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)

        depths = {"level_0": torch.FloatTensor(depth_0),
                  "level_1": torch.FloatTensor(depth_1),
                  "level_2": torch.FloatTensor(depth_2)}
        
        return depths

    def read_mask(self, filename):
        mask = cv2.imread(filename, 0) # (1200, 1600)
        if self.img_wh is None: # when we train it is None 
            mask = cv2.resize(mask, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST) # (600, 800)
            mask_0 = mask[44:556, 80:720] # (512, 640)
        else:
            mask_0 = cv2.resize(mask, self.img_wh,
                                interpolation=cv2.INTER_NEAREST)
        mask_1 = cv2.resize(mask_0, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST)
        mask_2 = cv2.resize(mask_1, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST)

        masks = {"level_0": torch.BoolTensor(mask_0),
                 "level_1": torch.BoolTensor(mask_1),
                 "level_2": torch.BoolTensor(mask_2)}

        return masks

    def read_semantic(self, filename):
        mask = cv2.imread(filename, 0) # (1200, 1600)
        if self.img_wh is None: # when we train it is None 
            mask = cv2.resize(mask, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST) # (600, 800)
            mask_0 = mask[44:556, 80:720] # (512, 640)
        else:
            mask_0 = cv2.resize(mask, self.img_wh,
                                interpolation=cv2.INTER_NEAREST)
        mask_1 = cv2.resize(mask_0, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST)
        mask_2 = cv2.resize(mask_1, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST)
        
        semantics = {"level_0": torch.ByteTensor(mask_0),
                     "level_1": torch.ByteTensor(mask_1),
                     "level_2": torch.ByteTensor(mask_2)}       # 8-bit integer (unsigned)

        return semantics   
    
    def read_planar_mask(self, filename):
        mask = cv2.imread(filename, 0) # (1200, 1600)
        if self.img_wh is None: # when we train it is None 
            mask = cv2.resize(mask, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST) # (600, 800)
            mask_0 = mask[44:556, 80:720] # (512, 640)
        else:
            mask_0 = cv2.resize(mask, self.img_wh,
                                interpolation=cv2.INTER_NEAREST)
        mask_1 = cv2.resize(mask_0, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST)
        mask_2 = cv2.resize(mask_1, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST)

        planar_masks = {"level_0": torch.BoolTensor(mask_0),
                        "level_1": torch.BoolTensor(mask_1),
                        "level_2": torch.BoolTensor(mask_2)}

        return planar_masks    
        
    def define_transforms(self):
        if self.split == 'train': # you can add augmentation here
            self.transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225]),
                                       ])
        else:
            self.transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225]),
                                       ])

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        sample = {}
        scan, light_idx, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]

        imgs = []
        proj_mats = [] # record proj mats between views
        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            if self.img_wh is None:
                img_filename = os.path.join(self.root_dir, f'Rectified/{scan}_train/rect_{vid+1:03d}_{light_idx}_r5000.png')
                mask_filename = os.path.join(self.root_dir, f'Depths/{scan}/depth_visual_{vid:04d}.png')
                depth_filename = os.path.join(self.root_dir, f'Depths/{scan}/depth_map_{vid:04d}.pfm')
                # QUESTION: do I want to infer different semantics for every image with different lighting conditions? 
                semantic_filename = os.path.join(self.root_dir, f'Semantics/{scan}_train/sem_{vid+1:03d}_3_r5000.png')        # one semantic map per viewpoint. 
                planar_mask_filename = os.path.join(self.root_dir, f'Semantics/{scan}_train/planar_{vid+1:03d}_3_r5000.png') 
            else:
                img_filename = os.path.join(self.root_dir,
                                f'Rectified/{scan}/rect_{vid+1:03d}_{light_idx}_r5000.png')

            img = Image.open(img_filename)
            if self.img_wh is not None:
                img = img.resize(self.img_wh, Image.BILINEAR)
            img = self.transform(img)
            imgs += [img]

            proj_mat_ls, depth_min = self.proj_mats[vid]

            if i == 0:  # reference view
                sample['init_depth_min'] = torch.FloatTensor([depth_min])
                if self.img_wh is None:
                    sample['masks'] = self.read_mask(mask_filename)
                    sample['depths'] = self.read_depth(depth_filename)
                    sample['semantics'] = self.read_semantic(depth_filename)
                    sample['planar_masks'] = self.read_planar_mask(depth_filename)                                        
                ref_proj_inv = torch.inverse(proj_mat_ls)
            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]

        imgs = torch.stack(imgs) # (V, 3, H, W)
        proj_mats = torch.stack(proj_mats)[:,:,:3] # (V-1, self.levels, 3, 4) from fine to coarse

        sample['imgs'] = imgs
        sample['proj_mats'] = proj_mats
        sample['depth_interval'] = torch.FloatTensor([self.depth_interval])
        sample['scan_vid'] = (scan, ref_view)

        return sample
