from .utils import read_pfm
from PIL import Image
import struct
import os
import numpy as np
from numpy import inf
import sys
import re
import cv2
from scipy import interpolate 

# STEP 1: get the filenames 
img_filename = os.path.join(self.root_dir, f'Rectified/{scan}_train/rect_{vid+1:03d}_{light_idx}_r5000.png')
mask_filename = os.path.join(self.root_dir, f'Depths/{scan}/depth_visual_{vid:04d}.png')
depth_filename = os.path.join(self.root_dir, f'Depths/{scan}/depth_map_{vid:04d}.pfm')


depth_dir = "/content/drive/MyDrive/MASTER_GEOMATICS/THESIS/ETH3D_dataset/facade_dslr_depth"
depth_image = os.path.join(depth_dir, 'DSC_0334.JPG')

# image_dir = "/content/drive/MyDrive/MASTER_GEOMATICS/THESIS/ETH3D_dataset/facade_dslr_undistorted/facade/images/dslr_images_undistorted/"
image_dir = "/content/drive/MyDrive/MASTER_GEOMATICS/THESIS/ETH3D_dataset/facade_dslr_distorted/"
original_image_path = os.path.join(image_dir, 'DSC_0334.JPG')

save_dir = "/content/drive/MyDrive/MASTER_GEOMATICS/THESIS/ETH3D_dataset/facade_final_depth/"
save_path1 = os.path.join(save_dir, 'DSC_0334_simple.pfm')
save_path2 = os.path.join(save_dir, 'DSC_0334_interpolated.pfm')
save_path_interpolated = os.path.join(save_dir, 'DSC_0334_interpolated.pfm')

save_mask_dir = "/content/drive/MyDrive/MASTER_GEOMATICS/THESIS/ETH3D_dataset/depth_masks/mask_lalala.png"
save_mask_path = os.path.join(save_mask_dir, 'mask_lalala.png')

mask_path = "/content/drive/MyDrive/MASTER_GEOMATICS/THESIS/ETH3D_dataset/"

# STEP 2: Process the ETH3D dataset to save the depth map in .pfm format. 

# Open corresponding image to get its dimensions. 
original_image = Image.open(original_image_path)
shape = original_image.size # (6048, 4032) --> (WIDTH, HEIGHT)  

# How to unpack with struct library: https://stackoverflow.com/questions/37093485/how-to-interpret-4-bytes-as-a-32-bit-float-using-python
with open(depth_image, "rb") as file:
    # Read the file contents
    file_contents = file.read()
    
    # Unpack the 4-byte floats. Floats are stored in little-endian format, so I use '<f' format specifier.  
    floats = struct.unpack("<{}f".format(len(file_contents) // 4), file_contents)  #  number of 4-byte floats in the file
    np_floats = np.array(floats, dtype= np.float32) 

    ## DEPTH MAP

    #  Float Map: <class 'numpy.ndarray'> float32
    np_floats[np_floats==inf] = 0   # change inf to 0 for pixels with no depth. 
    depth_map = np_floats = np_floats.reshape((shape[1], shape[0])) #  (Height, Width) --> (4032, 6048)
    
    ## Floating points rounding error due to storing them as binary representations: 
    ## Compute the maximum magnitude of the values to calculate the tolerance (epsilon)
    max_magnitude = np.max(np.abs(np_floats))  # the maximum value: 22.781841
    relative_tolerance = 1e-06
    epsilon = relative_tolerance * max_magnitude  # threshold: 2.278184127807617e-05

    ## MASK IMAGE: 

    ## FOLLOWING KWEA IMPLEMENTATION the mask is an numpy array of dtype=uint8, in .png format with cv2. 
    ## *** It will contain only [0 255] initially. Then in the Dataloader I will turn it into BoolTensor. 
    mask = np.full(shape=(np_floats.shape[0], np_floats.shape[1]), fill_value=255, dtype=np.uint8) # uint8 <class 'numpy.ndarray'> 
    mask[np.abs(np_floats) <= epsilon] = 0  
    cv2.imwrite("/content/drive/MyDrive/MASTER_GEOMATICS/THESIS/ETH3D_dataset/depth_masks/mask_lalalaq_cv2.png", mask) # Φαίνεται continuous αλλά είναι με missing values. 


    ## INTERPOLATED DEPTH map (MISSING POINTS ONLY)
    
    # Step 1: Generate coordinates of non-missing depth values
    coords = np.argwhere(depth_map != 0)      # returns the indices of the depth map that are non-zero. (non-missing)
    # Step 2: Extract non-missing depth values
    values = depth_map[depth_map != 0]        # reruns the values of the depth map that are non-zero. Δηλαδή γράψε στο array 'values' τις τιμές από το depth_map που δεν είναι 0. 
    # Step 3: Generate coordinates of all pixels in the depth map
    full_coords = np.indices(depth_map.shape).reshape(2, -1).T     # Returns an array representing the indices of a grid.  Reshape in unknown columns and rows = 2. # Πρακτικά προκύπτει [[0 0] [0 1] [0 2] [1 0] [1 1] [1 2]] για shape=(2,3)
    # Step 4: Perform interpolation using griddata
    interpolated_values = interpolate.griddata(coords, values, full_coords, method='nearest', fill_value = 0) # arguments: Data point coordinates, Data values, Points at which to interpolate data. Fill value for points outside the Convex Hull.  
    # Step 5: Create a new depth map with interpolated values
    interpolated_depth_map = np.reshape(interpolated_values, depth_map.shape)  # Must be 'float32' data type. It is! 
    
    
    ## FOR CONSISTENT AND ACCURATE VISUALIZATION (Επειδή κάποια softwares Βλέπε PfmPad adhere to the standard range [0,1].)
    
    ## Normalize the PFM to fall within [0, 1] 
    # min_value = np.min(interpolated_depth_map)
    # max_value = np.max(interpolated_depth_map)
    # normalized_data = (interpolated_depth_map - min_value) / (max_value - min_value)
    # interpolated_depth_map = normalized_data 


    ## WRITE the interpolated depth map in .pfm format (Portable FloatMap). View with PfmPad software. *** No need to keep the original .pfm since I will have the mask! ***   
    save_pfm(save_path2, interpolated_depth_map) #  SAVES THE INTERPOLATED DEPTH MAP
    # save_pfm(save_path1, depth_map) # SAVES THE ORIGINAL DEPTH MAP with missing values 
      
