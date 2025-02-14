import torch
from torch import nn

class SL1Loss(nn.Module):
    def __init__(self, levels=3):
        super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, inputs, targets, masks):
        loss = 0
        for l in range(self.levels):
            depth_pred_l = inputs[f'depth_{l}']
            depth_gt_l = targets[f'level_{l}']
            mask_l = masks[f'level_{l}']
            loss += self.loss(depth_pred_l[mask_l], depth_gt_l[mask_l]) * 2**(1-l)  # * loss weight of each level. For level = {0, 1, 2} we get the weights {2, 1, 1/2} respectively. Level_0 is the finest. 
        return loss

    
# New custom loss function    
class CustomLoss(nn.Module):  
    """ 
    # idea: keep the same loss SL1Loss and make it more sophisticated. 
    # New loss function consisting of 1) Depth-groundtruth loss, 2) Smoothness loss for planar regions    
    # Lgt = | depth_groundtruth[valid_mask] - depth_predicted[valid_mask] |  . This loss term measures the difference in depth maps between prediction and ground truth. --> L1 distance.    
    # Lplanar = valid_mask * planar_mask * | second order derivative of depth map | * exp(-| second order derivative of semantic map |)
    # Total_Loss = Lgt + Lplanar 
    """ 
    
    def __init__(self, levels=3):  
        super(CustomLoss, self).__init__()  # will call the __init__ function as seen by the parent class of CustomLoss 
        self.levels = levels               
        
    def forward(self, inputs, targets, masks, semantics, planar_masks):      
        
        loss = 0
        for l in range(self.levels):
            depth_pred_l = inputs[f'depth_{l}']
            depth_gt_l = targets[f'level_{l}']
            mask_l = masks[f'level_{l}']
            semantic_l = semantics[f'level_{l}']
            planar_mask_l = planar_masks[f'level_{l}']
            
            depth1 = depth_pred_l[mask_l]
            depth2 = depth_gt_l[mask_l]
                        
            # depth-groundtruth loss --> CasMVSNet (DDL-MVS combines ground-truth and smoothness term in one loss function)
            loss += 2**(1-l) * nn.SmoothL1Loss(depth1, depth2, reduction="mean")  # reduction='mean': the sum of the output will be divided by the number of elements in the output            
            MAIN_LOSS = loss
            
            # based on DDL-MVS
            # semantic smoothness loss to encourage local smoothness for planar regions WITHOUT depth discontinuities (penalize second-order depth variations)
            # --> dimensions of the predicted depth is (B, h, w)                           
            laplacian_depthy = torch.abs(2*depth_pred_l[:,1:-1,:] - depth_pred_l[:,:-2,:] - depth_pred_l[:,2:,:])   # [:,1:-1,:] discards the first and last row 
            laplacian_depthx = torch.abs(2*depth_pred_l[:,:,1:-1] - depth_pred_l[:,:,:-2] - depth_pred_l[:,:,2:])   # [:,:,1:-1] discards the first and last col    
            
            # apply the Positive Laplacian operator (takes out outward edges) to the semantic map  
            laplacian_semanticy = torch.abs(2*semantic_l[:,1:-1,:] - semantic_l[:,:-2,:] - semantic_l[:,2:,:])
            laplacian_semanticx = torch.abs(2*semantic_l[:,:,1:-1] - semantic_l[:,:,:-2] - semantic_l[:,:,2:])    
            
            # Applying the Laplacian operator to the semantic map to identify the edges and then use it as following guarantees that only non-boundary regions contribute to the Loss.            
            # multiply with the valid and planar mask to filter out invalid and non-planar pixels respectively
            # NOTE: the Laplacian operator can produce both positive and negative values in the response image, thus the absolute values are taken.
            # NOTE: in the response image, edges will have higher values while non-edge pixels will have values close to zero.
            # QUESTION: Should I turn the laplacian response to a binary image with 1s and 0s for edges, non-edges and then use BETA to control the switch on and off the contribution to the loss? -> DDL-MVS uses BETA = -20 which turns exp(-20 * 0) = exp(0) = 1 and exp(-20 * 1) = 0.
            tv_h = (planar_mask_l[:,:,1:-1,:] * mask_l[:,:,1:-1,:] * torch.abs(laplacian_depthy) * torch.exp(-torch.abs(laplacian_semanticy[:,:,1:-1,:]))).sum() # mask_l indicates the valid pixels, planar_mask_l indicates the planar pixels 
            tv_w = (planar_mask_l[:,:,:,1:-1] * mask_l[:,:,:,1:-1] * torch.abs(laplacian_depthx) * torch.exp(-torch.abs(laplacian_semanticx[:,:,:,1:-1]))).sum() 
            
            SMOOTH_LOSS += 2**(1-l) * (tv_h + tv_w)/len(depth1) # divided by the number of pixels of the image. 
                    
                                                    
        # total loss 
        loss = 2 * loss  + 1 * SMOOTH_LOSS
        
        print("TOTAL LOSS ",loss," MAIN LOSS ",MAIN_LOSS," SMOOTH_LOSS ",SMOOTH_LOSS)
        
        return loss          
    
# Dictionary with options for loss functions    
loss_dict = {'sl1': SL1Loss, 'custom_loss': CustomLoss}
