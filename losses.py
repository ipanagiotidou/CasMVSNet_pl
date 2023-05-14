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
            loss += self.loss(depth_pred_l[mask_l], depth_gt_l[mask_l]) * 2**(1-l)  # * loss weight of each stage
        return loss

    
# New custom loss function    
class CustomLoss():  # nn.Module
    """ 
    # I: idea: keep the same loss SL1Loss and make it more sophisticated. 
    # I: --> New loss function consisting of 1) Depth-groundtruth loss, 2) Smoothness loss, 3) Planarity loss.
    # I: 
    # I: --> Lgt = | depth_groundtruth - depth_predicted |  . This loss term measures the difference in depth maps between prediction and ground truth. --> L1 distance. 
    # I: --> Lsmooth = |  first order detivative of depth map | * exp(-| first order derivative of semantic map |) και 
    # I: --> Lplanar = | second order derivative of depth map | * exp(-| ????? order derivative of semantic map |)
    """ 
    
    def __init__(self, levels=3,
                 depth_pred,
                 depth_gt,
                 mask,
                 semantic_map,                                  
                 ): # example of type 'depth_gt: Dict[str, torch.Tensor]'
        
        # super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.SmoothL1Loss(reduction='mean') 
    

    # I: takes as input the 'Smooth' and 'Planar' Masks indicating which pixels belong to which object category. Then enforce a priori knowledge of these classes to smooth the depth. 
    def forward(self, inputs, targets, masks): 
        
        # depth-groundtruth loss --> CasMVSNet 
        loss = 0
        for l in range(self.levels):
            depth_pred_l = inputs[f'depth_{l}']
            depth_gt_l = targets[f'level_{l}']
            mask_l = masks[f'level_{l}']
            # I: smooth_mask_l stands for the mask indicating with 1s the pixels that belong to smooth classes
            # I: planar_mask_l stands for the mask indicating with 1s the pixels that belong to planar classes
            # I: the first term is for the SMOOTH areas and the second for the PLANAR AREAS 
            loss += self.loss(depth_pred_l[mask_l], depth_gt_l[mask_l]) * 2**(1-l)
        main_loss = loss 
        
        # NOTE: I probably need to do the below for all stages 
        
        # semantic smoothness loss (for planar and non-planar surfaces) --> DDL-MVS (for Nail stage l=0 represents the finest input and output resolution)
        # I: semantic-aware smoothness loss term to penalize first-order and second-order depth varations in non-boundary regions for non-planar and planar surfaces respectively. 
        # I: Only the non-boundary regions contribute to the Loss. The boundary regions do not contribute to the loss. 
        
        # laplacian is the second-order derivative (for x and y) 
        # --> check Unsupervised MVSNet to undestand the gradients: https://github.com/ipanagiotidou/unsup_mvs/blob/master/code/unsup_mvsnet/model.py 
        # example για Tensor με dimensions: [ ..? , y = height, x = width, ..? ]  
        # για το gradient του image έχουμε: 
        # --> άρα για y = height έχουμε gradient_y(img): return img[:, :-1, :  , :] - img[:, 1:, : , :]
        # --> και για x = width  έχουμε gradient_x(img): return img[:, :  , :-1, :] - img[:, : , 1:, :]     
        
        laplacian_depthy = torch.abs(2*refined_depth[f'stage_{l}'][:,:,1:-1,:] - refined_depth[f'stage_{l}'][:,:,:-2,:] - refined_depth[f'stage_{l}'][:,:,2:,:])
        laplacian_depthx = torch.abs(2*refined_depth[f'stage_{l}'][:,:,:,1:-1] - refined_depth[f'stage_{l}'][:,:,:,:-2] - refined_depth[f'stage_{l}'][:,:,:,2:])

        BETA = -20.  # turns exp(-20 * 0) = exp(0) = 1 and exp(-20 * 1) = 0 
        tv_h = (mask_l[:,:,1:-1,:]*torch.abs(laplacian_depthy)*torch.exp(BETA*edges_est[f'stage_{l}'][:,:,1:-1,:])).sum()
        tv_w = (mask_l[:,:,:,1:-1]*torch.abs(laplacian_depthx)*torch.exp(BETA*edges_est[f'stage_{l}'][:,:,:,1:-1])).sum()
       
        TV2LOSS = 2.5*(tv_h + tv_w)/len(depth1) # 2500
        
        
        
        # I: semantic-aware planar loss term to penalize second-order depth variations in non-boundary regions
        
        
        
        
        
        
            
        # total loss
        loss = 2*loss  + ... 
        
        print("TOTAL LOSS ",loss," MAIN LOSS ",main_loss," EDGEL2SIM ",EDGEL2SIM, " TV2LOSS ",TV2LOSS," BIMODAL_LOSS ",BIMODAL_LOSS)
        
        return loss          
    
    
loss_dict = {'sl1': SL1Loss, 'custom_loss': CustomLoss}
