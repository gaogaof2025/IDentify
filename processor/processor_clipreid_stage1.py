import logging
import os
import random
# from processor.processor_clipreid_stage2 import add_noise
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional

from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss


def show_img(img):
    torchvision.transforms.functional.to_pil_image(img).show()

def do_train_stage1(cfg,
             model,
             train_loader_stage1,
             optimizer,
             scheduler,
             local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD 

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    image_features = []
    image_features_original = []
    labels = []
    with torch.no_grad():
        for n_iter, (imgs, pids, camids, clothes_ids, cloth, _cloth) in enumerate(train_loader_stage1):
            cloth = torch.stack(cloth).to(device)
            cloth_mask = (cloth > 0).float()
            cloth_mask = cloth_mask.repeat(1, 3, 1, 1)


            img = imgs.to(device)
            target = pids.to(device)

            img = img * (1 - cloth_mask) + cloth * cloth_mask
        
        
            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image = True)
                # feat_original = model(img_original, target, get_image=True)
                for i, img_feat in zip(target, image_feature):
                    labels.append(i)
                    image_features.append(img_feat.cpu())
                    
                    
        labels_list = torch.stack(labels, dim=0).cuda() #N
        image_features_list = torch.stack(image_features, dim=0).cuda()
        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, image_features
    tcl_loss_fn = TextConsistencyLoss()
    lowest_loss = np.inf
    early_stop_count = 0
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter+1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i*batch:(i+1)* batch]
            else:
                b_list = iter_list[i*batch:num_image]
            
            target = labels_list[b_list]
            image_features = image_features_list[b_list]
            # img_feat_original = img_feat_original_list[b_list]
            with amp.autocast(enabled=True):
                text_features = model(label = target, get_text = True)
            loss_i2t = xent(image_features, text_features, target, target)
            loss_t2i = xent(text_features, image_features, target, target)
            
            loss_tcl = tcl_loss_fn(image_features,text_features,target)
            loss = loss_i2t + loss_t2i + loss_tcl*0.5

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(train_loader_stage1),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))
        if loss_meter.avg < lowest_loss:
            lowest_loss = loss_meter.avg
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count == 10:
                logger.info("Early stop at epoch {}".format(epoch))
                break

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextConsistencyLoss(nn.Module):
    """
    TCL for CC-ReID
    """
