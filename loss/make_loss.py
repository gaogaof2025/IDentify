# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss, ContrastiveCenterLoss
from .clothes_based_adversarial_loss import ClothesBasedAdversarialLoss

from torch import nn

def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = ContrastiveCenterLoss(num_classes=num_classes, feat_dim=feat_dim)  # center loss
    # center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    cla_criterion = ClothesBasedAdversarialLoss(scale=16, epsilon=0.1)
    # cmsm_criterion = Loss_CMSM()
    criterion_stripe = nn.CrossEntropyLoss()
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        # def loss_func(score, feat, target, feat_2, cls_score_2, feat_3, cls_score_3, text_feat, i2tscore = None, positive_mask= None, cls_cloth_score = None):
        def loss_func(score, feat, target, text_feat, i2tscore=None,positive_mask=None, cls_cloth_score=None, target_mask = None, attn_weights= None):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS) 
                    else:   
                        TRI_LOSS = triplet(feat, target)[0]
                    

                    # cla_loss = cla_criterion(cls_cloth_score, target, positive_mask)
                    loss = (cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS +
                            cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                            # + cfg.MODEL.CAL_LOSS_WEIGHT * cla_loss
                            )

                    if i2tscore != None:
                        I2TLOSS = xent(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss
                        
                    return loss
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                            TRI_LOSS = sum(TRI_LOSS)
                    else:
                            TRI_LOSS = triplet(feat, target)[0]
                    cloth_mask_down = F.adaptive_avg_pool2d(target_mask, (24, 12))
                    B, N_patches, _ = attn_weights.shape
                    cloth_mask_down = cloth_mask_down.view(B, N_patches, 1)
                    loss_attn_mask = F.mse_loss(attn_weights, cloth_mask_down)

                    cla_loss = cla_criterion(cls_cloth_score, target, positive_mask)

                    # id_2 = F.cross_entropy(cls_score_2, target)
                    # id_3 = F.cross_entropy(cls_score_3, target)
                    # tri_2 = triplet(feat_2, target)[0]
                    # tri_3 = triplet(feat_3, target)[0]

                    # id = F.cross_entropy(cls_score_2, target) + F.cross_entropy(cls_score_3, target)
                    # tri = triplet(feat_2, target)[0] + triplet(feat_3, target)[0]
                    #
                    # ID_LOSS = ID_LOSS + id
                    # TRI_LOSS = TRI_LOSS + tri

                    # ID_LOSS = ID_LOSS
                    # TRI_LOSS = TRI_LOSS

                    # loss = (cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS  +
                    #         cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    #         + cfg.MODEL.CAL_LOSS_WEIGHT * cla_loss + cfg.MODEL.ID_LOSS * (id + tri)
                    #         )
                    loss = (cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS + 0 * cla_loss+loss_attn_mask)

                    if i2tscore != None:
                        I2TLOSS = F.cross_entropy(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss


                    return {
                        'LOSS': loss,
                        'ID_LOSS': ID_LOSS,
                        'TRI_LOSS': TRI_LOSS,
                        'CLA_LOSS': cla_loss
                        # 'HW' : CrossEntropy_Loss
                    }
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


