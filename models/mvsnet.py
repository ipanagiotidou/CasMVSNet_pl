from einops import reduce, rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *

class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self, norm_act=InPlaceABN):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

        self.conv2 = nn.Sequential( 
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        # to reduce channel size of the outputs from FPN
        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, 
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        conv0 = self.conv0(x) # (B, 8, H, W)
        conv1 = self.conv1(conv0) # (B, 16, H//2, W//2)
        conv2 = self.conv2(conv1) # (B, 32, H//4, W//4)
        feat2 = self.toplayer(conv2) # (B, 32, H//4, W//4)
        feat1 = self._upsample_add(feat2, self.lat1(conv1)) # (B, 32, H//2, W//2)
        feat0 = self._upsample_add(feat1, self.lat0(conv0)) # (B, 32, H, W)

        # reduce output channels
        feat1 = self.smooth1(feat1) # (B, 16, H//2, W//2)
        feat0 = self.smooth0(feat0) # (B, 8, H, W)

        feats = {"level_0": feat0,
                 "level_1": feat1,
                 "level_2": feat2}

        return feats


class CostRegNet(nn.Module):
    def __init__(self, in_channels, norm_act=InPlaceABN):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(8))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        x = self.prob(x)
        return x


class CascadeMVSNet(nn.Module):
    # I: the depth interval in CasMVSNet is set to [4, 2, 1] times as the interval of MVSNet (which is a fixed number). 
    # I: depth interval for the finest level  = 2.65 mm (from opt.py)
    def __init__(self, n_depths=[8, 32, 48],
                       interval_ratios=[1, 2, 4],
                       num_groups=1,
                       norm_act=InPlaceABN):
        super(CascadeMVSNet, self).__init__()
        self.levels = 3 # 3 depth levels
        self.n_depths = n_depths
        self.interval_ratios = interval_ratios
        self.G = num_groups # number of groups in groupwise correlation
        self.feature = FeatureNet(norm_act)
        for l in range(self.levels):
            if self.G > 1:
                cost_reg_l = CostRegNet(self.G, norm_act)
            else:
                cost_reg_l = CostRegNet(8*2**l, norm_act) # I: in_channels = 8*2**l όπου για l= level = {0,1,2} έχουμε {8, 16, 32} αντίστοιχα.
            setattr(self, f'cost_reg_{l}', cost_reg_l)    # I: 'cost_reg_0', etc. becomes an attribute of CascadeMVSNet. Retrieves later with getattr(self, f'cost_reg_{l}') and pass it to def predict_depth.

    def predict_depth(self, feats, proj_mats, depth_values, cost_reg):
        # feats: (B, V, C, H, W)
        # proj_mats: (B, V-1, 3, 4)
        # depth_values: (B, D, H, W)    # I: for each Batch we have one volume of dimension (D, H, W) where D is the depth dimension and (H, W) the spatial dim.
        # cost_reg: nn.Module of input (B, C, D, h, w) and output (B, 1, D, h, w)    # I: outputs one probability volume, B is the batch 
        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]   # I: number of depth hypothesis

        ref_feats, src_feats = feats[:, 0], feats[:, 1:]
        src_feats = rearrange(src_feats, 'b vm1 c h w -> vm1 b c h w') # (V-1, B, C, h, w)
        proj_mats = rearrange(proj_mats, 'b vm1 x y -> vm1 b x y') # (V-1, B, 3, 4)

        ref_volume = rearrange(ref_feats, 'b c h w -> b c 1 h w')
        ref_volume = repeat(ref_volume, 'b c 1 h w -> b c d h w', d=D) # (B, C, D, h, w)
        if self.G == 1:
            volume_sum = ref_volume
            volume_sq_sum = ref_volume ** 2
        else:
            ref_volume = ref_volume.view(B, self.G, C//self.G, *ref_volume.shape[-3:])
            volume_sum = 0
        del ref_feats

        for src_feat, proj_mat in zip(src_feats, proj_mats):
            warped_volume = homo_warp(src_feat, proj_mat, depth_values)
            warped_volume = warped_volume.to(ref_volume.dtype)
            if self.G == 1:
                if self.training:
                    volume_sum = volume_sum + warped_volume
                    volume_sq_sum = volume_sq_sum + warped_volume ** 2
                else:
                    volume_sum += warped_volume
                    volume_sq_sum += warped_volume.pow_(2)
            else:
                warped_volume = warped_volume.view_as(ref_volume)
                if self.training:
                    volume_sum = volume_sum + warped_volume # (B, G, C//G, D, h, w)
                else:
                    volume_sum += warped_volume
            del warped_volume, src_feat, proj_mat
        del src_feats, proj_mats
        # aggregate multiple feature volumes by variance
        if self.G == 1:
            volume_variance = volume_sq_sum.div_(V).sub_(volume_sum.div_(V).pow_(2))
            del volume_sq_sum, volume_sum
        else:
            volume_variance = reduce(volume_sum*ref_volume,
                                     'b g c d h w -> b g d h w', 'mean').div_(V-1) # (B, G, D, h, w)
            del volume_sum, ref_volume
        
        cost_reg = rearrange(cost_reg(volume_variance), 'b 1 d h w -> b d h w')
        prob_volume = F.softmax(cost_reg, 1) # (B, D, h, w)
        del cost_reg
        depth = depth_regression(prob_volume, depth_values)
        
        with torch.no_grad():
            # sum probability of 4 consecutive depth indices
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1),
                                                      pad=(0, 0, 0, 0, 1, 2)),
                                                (4, 1, 1), stride=1).squeeze(1) # (B, D, h, w)
            # find the (rounded) index that is the final prediction
            depth_index = depth_regression(prob_volume,
                                           torch.arange(D,
                                                        device=prob_volume.device,
                                                        dtype=prob_volume.dtype)
                                          ).long() # (B, h, w)
            depth_index = torch.clamp(depth_index, 0, D-1)
            # the confidence is the 4-sum probability at this index
            confidence = torch.gather(prob_volume_sum4, 1, 
                                      depth_index.unsqueeze(1)).squeeze(1) # (B, h, w)

        return depth, confidence

    def forward(self, imgs, proj_mats, init_depth_min, depth_interval):
        # imgs: (B, V, 3, H, W)
        # proj_mats: (B, V-1, self.levels, 3, 4) from fine to coarse
        # init_depth_min, depth_interval: (B) or float
        B, V, _, H, W = imgs.shape
        results = {}

        imgs = imgs.reshape(B*V, 3, H, W)
        feats = self.feature(imgs) # (B*V, 8, H, W), (B*V, 16, H//2, W//2), (B*V, 32, H//4, W//4)
        
        for l in reversed(range(self.levels)): # (2, 1, 0)
            feats_l = feats[f"level_{l}"] # (B*V, C, h, w)
            feats_l = feats_l.view(B, V, *feats_l.shape[1:]) # (B, V, C, h, w)
            proj_mats_l = proj_mats[:, :, l] # (B, V-1, 3, 4)
            depth_interval_l = depth_interval * self.interval_ratios[l]
            D = self.n_depths[l]
            if l == self.levels-1: # coarsest level
                h, w = feats_l.shape[-2:]
                if isinstance(init_depth_min, float):     # I: returns True if init_depth_min is a float
                    depth_values = init_depth_min + depth_interval_l * \
                                   torch.arange(0, D,
                                                device=imgs.device,
                                                dtype=imgs.dtype) # (D)   # probably 1d Tensor for each pixel # I: all hypothesis depths (for each pixel). Values from [depth_min, depth_max) with Step = depth_interval.
                    depth_values = rearrange(depth_values, 'd -> 1 d 1 1')
                    depth_values = repeat(depth_values, '1 d 1 1 -> b d h w', b=B, h=h, w=w)  # B is the batch. Repeat for each batch: each depth hypothesis value is repeated along the spatial dimension. D acts as the number of channels.
                else:
                    # I: if not the coarsest level, it should form the depth hypothesis based on the previous level's depth prediction 
                    depth_values = init_depth_min + depth_interval_l * \
                                   rearrange(torch.arange(0, D,
                                                          device=imgs.device,
                                                          dtype=imgs.dtype),
                                             'd -> 1 d') # (B, D)
                    depth_values = rearrange(depth_values, 'b d -> b d 1 1')
                    depth_values = repeat(depth_values, 'b d 1 1 -> b d h w', h=h, w=w)
                    
            else:  # I: if not the coarsest level 
                # First the network predicts the coarsest depth map, it then upscales it using interpolation to get the new depth values that will constitute a starting point for next level.
                # To do so it detaches the previous level from the graph and once she gets the depth values (which are now not from depth min to depth max, but are more refined and closer to the actual prediction) she deletes the new vatriable. 
                # The depth values are passed to the 'predict_depth' function. 
                depth_lm1 = depth_l.detach() # Detaches the depth of previous level # Has already returned it 
                depth_lm1 = F.interpolate(rearrange(depth_lm1, 'b h w -> b 1 h w'),
                                          scale_factor=2, mode='bilinear',
                                          align_corners=True) # (B, 1, h, w)  # upscales by 2 the predicted depth map using interpolation 
                depth_values = get_depth_values(depth_lm1, D, depth_interval_l)   # I: get_depth_values(current_depth, n_depths, depth_interval) # each level has its own n_depths[l], depth_interval and current depth is the depth of the previous level 
                del depth_lm1
            
            # I: when coarse level (l=2) the first time, we pass the B volumes with the initial depth hypothesis (depth_values from depth min to depth max).
            # I: when not the coarsest level, we pass the depth_values from the predicted depth map of the coarsest level to act as the basis to find the new depth prediction for the new level.
            
            depth_l, confidence_l = self.predict_depth(feats_l, proj_mats_l, depth_values,
                                                       getattr(self, f'cost_reg_{l}'))
            del feats_l, proj_mats_l, depth_values
            results[f"depth_{l}"] = depth_l
            results[f"confidence_{l}"] = confidence_l
            

        return results
