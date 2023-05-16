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
    # I: --> New loss function consisting of 1) Depth-groundtruth loss, 2) Smoothness loss, 3) Planarity loss, with the last two being mutually exclusive. OR simply have the Planarity loss. 
    # I: 
    # I: --> Lgt = | depth_groundtruth - depth_predicted |  . This loss term measures the difference in depth maps between prediction and ground truth. --> L1 distance. 
    # I: --> Lsmooth = |  first order detivative of depth map | * exp(-| first order derivative of semantic map |) και 
    # I: --> Lplanar = | second order derivative of depth map | * exp(-| ????? order derivative of semantic map |)
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
            semantic_l = semantic_maps
            planar_mask = planar_masks
            
            # TO DO: add the semantic map 
            
            # depth-groundtruth loss --> CasMVSNet (DDL-MVS combines ground-truth and smoothness term in one loss function)
            loss += self.loss(depth_pred_l[mask_l], depth_gt_l[mask_l]) * 2**(1-l)
            main_loss = loss 
            
            # semantic smoothness loss to encourage local smoothness for planar regions WITHOUT depth discontinuities (penalize second-order depth variations)
            # --> dimensions of the predicted depth is (B, h, w) --> index and slice the height and width dimensions.
            # --> TO DO: resize on the fly the planar mask to match the size of the output depth map of levels 1, 2.             
            # DDL-MVS 
            laplacian_depthy = torch.abs(2*depth_pred_l[:,1:-1,:] - depth_pred_l[:,:-2,:] - depth_pred_l[:,2:,:])   # [:,1:-1,:] discard the first and last row 
            laplacian_depthx = torch.abs(2*depth_pred_l[:,:,1:-1] - depth_pred_l[:,:,:-2] - depth_pred_l[:,:,2:])   # [:,:,1:-1] discard the first and last col
            
            # --> TO DO: apply the Laplacian operator to the semantic map  
            # NOTE: the semantic map can be turned to a binary indicating planar or non-planar areas. This will act as a mask to filter out pixels that we don't want them to contribute to the Loss. (P = 1 - S)
            laplacian_semanticy = torch.abs(2*semantic_map[:,1:-1,:] - semantic_map[:,:-2,:] - semantic_map[:,2:,:])
            laplacian_semanticx = torch.abs(2*semantic_map[:,:,1:-1] - semantic_map[:,:,:-2] - semantic_map[:,:,2:])
            
            # Applying the Laplacian to the semantic map you guarantee that only non-boundary regions contribute to the Loss.
            BETA = -20.  # turns exp(-20 * 0) = exp(0) = 1 and exp(-20 * 1) = 0 
            tv_h = (laplace_depth_mask_l[:,:,1:-1,:]*torch.abs(laplacian_depthy)*torch.exp(BETA*edges_est[f'stage_{l}'][:,:,1:-1,:])).sum()
            tv_w = (laplace_depth_mask_l[:,:,:,1:-1]*torch.abs(laplacian_depthx)*torch.exp(BETA*edges_est[f'stage_{l}'][:,:,:,1:-1])).sum()

            # multiply the loss term with the planar_mask to filter out pixels with no need to contribute to the Loss. 
            TV2LOSS = 2.5*(tv_h + tv_w)/len(depth1) # 2500
            
        
                                        
            
        # total loss
        loss = 2*loss  + ... 
        
        print("TOTAL LOSS ",loss," MAIN LOSS ",main_loss," EDGEL2SIM ",EDGEL2SIM, " TV2LOSS ",TV2LOSS," BIMODAL_LOSS ",BIMODAL_LOSS)
        
        return loss          
    
    
loss_dict = {'sl1': SL1Loss, 'custom_loss': CustomLoss}
