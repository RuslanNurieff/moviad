from .model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from .loss import SSIM, FocalLoss

import torch
import torch.nn as nn

import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class DRAEM(nn.Module):

    DEFAULT_PARAMETERS = {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.0001,
    }

    def __init__(self, device):
        super(DRAEM, self).__init__()
        self.device = device
        self.model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        self.model = self.model.to(self.device)
        self.model.apply(weights_init)

        self.model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        self.model_seg = self.model_seg.to(self.device)
        self.model_seg.apply(weights_init)
        
        # self.training = True

        self.loss_l2 = torch.nn.modules.loss.MSELoss()
        self.loss_ssim = SSIM()
        self.loss_focal = FocalLoss()

    def train(self, *args, **kwargs):
        self.model.train()
        self.model_seg.train()
        return super().train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        self.model.eval()
        self.model_seg.eval()
        return super().eval(*args, **kwargs)

    
    def forward(self, batch):
        if self.training:
            gray_batch = batch[0].to(self.device)
            aug_gray_batch = batch[1].to(self.device)
            anomaly_mask = batch[2].to(self.device)

            gray_rec = self.model(aug_gray_batch)
            joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

            out_mask = self.model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            l2_loss = self.loss_l2(gray_rec,gray_batch)
            ssim_loss = self.loss_ssim(gray_rec, gray_batch)

            segment_loss = self.loss_focal(out_mask_sm, anomaly_mask)
            # print(f"L2 Loss: {l2_loss}, SSIM Loss: {ssim_loss}, Segment Loss: {segment_loss}")
            loss = l2_loss + ssim_loss + segment_loss

            return loss
        else:
            out_mask_cv, image_score = self.predict(batch)
            return out_mask_cv, image_score


    def predict(self, gray_batch):
        with torch.no_grad():
            gray_rec = self.model(gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

            out_mask = self.model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            out_mask_cv = out_mask_sm[:, 1:2,: ,:].detach().cpu().numpy() #(1, 256, 256)
            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                               padding=21 // 2).cpu().detach().numpy()
            image_score = np.max(out_mask_averaged, axis=(1, 2, 3))

        return out_mask_cv, image_score
        