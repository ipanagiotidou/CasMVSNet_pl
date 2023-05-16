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
class CustomLoss():  # nn.Module
    """ 
    # I: idea: keep the same loss SL1Loss and make it more sophisticated. 
    # I: --> New loss function consisting of 1) Depth-groundtruth loss, 2) Smoothness loss for planar regions    
    # I: --> Lgt = | depth_groundtruth[valid_mask] - depth_predicted[valid_mask] |  . This loss term measures the difference in depth maps between prediction and ground truth. --> L1 distance.    
    # I: --> Lplanar = valid_mask * planar_mask * | second order derivative of depth map | * exp(-| second order derivative of semantic map |)
    """ 
    
    def __init__(self, levels=3):  # super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.SmoothL1Loss(reduction='mean') 
        
        
    def forward(self, inputs, targets, masks, semantic_maps, planar_masks):      
        
        loss = 0
        for l in range(self.levels):
            depth_pred_l = inputs[f'depth_{l}']
            depth_gt_l = targets[f'level_{l}']
            mask_l = masks[f'level_{l}']
            semantic_map_l = semantic_maps[f'level_{l}']
            planar_mask_l = planar_masks[f'level_{l}']
                        
            # depth-groundtruth loss --> CasMVSNet (DDL-MVS combines ground-truth and smoothness term in one loss function)
            main_loss += self.loss(depth_pred_l[mask_l], depth_gt_l[mask_l]) * 2**(1-l)
            
            
            # semantic smoothness loss to encourage local smoothness for planar regions WITHOUT depth discontinuities (penalize second-order depth variations)
            # --> dimensions of the predicted depth is (B, h, w)              
            # based on DDL-MVS 
            laplacian_depthy = torch.abs(2*depth_pred_l[:,1:-1,:] - depth_pred_l[:,:-2,:] - depth_pred_l[:,2:,:])   # [:,1:-1,:] discards the first and last row 
            laplacian_depthx = torch.abs(2*depth_pred_l[:,:,1:-1] - depth_pred_l[:,:,:-2] - depth_pred_l[:,:,2:])   # [:,:,1:-1] discards the first and last col
            
            # apply the Positive Laplacian operator (takes out outward edges) to the semantic map  
            # NOTE: the planar_mask indicates planar regions and filters out pixels with no need to contribute to the Loss.
            laplacian_semanticy = torch.abs(2*semantic_map_l[:,1:-1,:] - semantic_map_l[:,:-2,:] - semantic_map_l[:,2:,:])
            laplacian_semanticx = torch.abs(2*semantic_map_l[:,:,1:-1] - semantic_map_l[:,:,:-2] - semantic_map_l[:,:,2:])
            
            # Applying the Laplacian operator to the semantic map to identify the edges and then use it as following guarantees that only non-boundary regions contribute to the Loss.            
            # multiply the loss term with the mask to filter out invalid pixels (pixels for which we don't know their depth ground truth value). 
            # multiply the loss term with the planar_mask to filter out pixels that do not belong to planar regions, therefore there is no need to contribute to the Loss. 
            # --> Question: what is the response of the Laplacian operator and how can I use it to switch on or off the loss. --> DDL-MVS uses BETA = -20 which turns exp(-20 * 0) = exp(0) = 1 and exp(-20 * 1) = 0.
            tv_h = (planar_mask_l[:,:,1:-1,:] * mask_l[:,:,1:-1,:] * torch.abs(laplacian_depthy) * torch.exp(laplacian_semanticy[:,:,1:-1,:])).sum() # mask_l indicates the valid pixels, planar_mask_l indicates the planar pixels 
            tv_w = (planar_mask_l[:,:,:,1:-1] * mask_l[:,:,:,1:-1] * torch.abs(laplacian_depthx) * torch.exp(laplacian_semanticx[:,:,:,1:-1])).sum() 
            
            TV2LOSS = 2.5*(tv_h + tv_w)/len(depth1) # 2500
            
        
                                        
            
        # total loss
        loss = 2*main_loss  + ... 
        
        print("TOTAL LOSS ",loss," MAIN LOSS ",main_loss," EDGEL2SIM ",EDGEL2SIM, " TV2LOSS ",TV2LOSS," BIMODAL_LOSS ",BIMODAL_LOSS)
        
        return loss          
    
    
loss_dict = {'sl1': SL1Loss, 'custom_loss': CustomLoss}
