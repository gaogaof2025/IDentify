import logging
import math
import os
import time
import torch
import torch.nn as nn
import torchvision.transforms.functional

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss

from torch.utils.tensorboard import SummaryWriter


def show_image(image):
    torchvision.transforms.functional.to_pil_image(image).show()


def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank, pid2clothes, num_query_same, num_query_diff):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE
    writer = SummaryWriter(log_dir="./tb_log")
    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval( cfg.DATASETS.NAMES,num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,num_query_same=num_query_same, num_query_diff=num_query_diff)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    # train
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes-batch* (num_classes//batch)
    if left != 0 :
        i_ter = i_ter+1
    text_features_list = []
    # text_features_original_list = []
    with torch.no_grad():
        for i in range(i_ter):
            if i+1 != i_ter:
                l_list = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list = torch.arange(i*batch, num_classes)
            with amp.autocast(enabled=True):
                text_feat = model(label = l_list, get_text = True)
            text_features_list.append(text_feat.cpu())
            # text_features_original_list.append(text_feat_original.cpu())
            
        text_features = torch.cat(text_features_list, 0).cuda()
        # text_features_original = torch.cat(text_features_original_list, 0).cuda()
    best_map = 0
    best_rank1 = 0
    early_stop_count = 0
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()


        model.train()
        logger.info("Current Epoch replace_ratio {}".format(linear_growth(epoch, 20)))
        for n_iter, (imgs, pids, camids, clothes_ids, cloth, _cloth) in enumerate(train_loader_stage2):

            if n_iter == 1000:
                break

            optimizer.zero_grad()
            optimizer_center.zero_grad()

            pos_mask = pid2clothes[pids.cpu()].cuda()
            pos_mask = pos_mask.float().cuda()

            img = imgs.to(device)
            cloth = torch.stack(cloth).to(device)
            cloth_mask = (cloth > 0).float()
            cloth_mask = cloth_mask.repeat(1, 3, 1, 1)
            result = img * (1 - cloth_mask) + cloth * cloth_mask
            

            _cloth = torch.stack(_cloth).to(device)
            target_attn = (_cloth > 0).float()

            target = pids.to(device)

            with amp.autocast(enabled=True):
                score, feat, image_features, cls_cloth_score, attn_weights = model(x = img, label = target, noise_img= result)

                # score, feat, image_features, cls_cloth_score = model(x = img, label = target, noise_img= None)
                logits = image_features @ text_features.t()
                total_loss = loss_fn(score, feat, target, text_features, logits, pos_mask, cls_cloth_score, target_attn, attn_weights)

            loss = total_loss['LOSS']
            loss_str = ""
            for key, value in total_loss.items():
                loss_str += "{}: {:.3f}, ".format(key, value.item())

            # 计算错误分类的掩码
            misclassified_mask = logits.max(1)[1] != target

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)



            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] {} Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_str, acc_meter.avg, scheduler.get_lr()[0]))
                writer.add_scalar('Loss/train', loss_meter.avg, epoch * len(train_loader_stage2) + n_iter)
                writer.add_scalar('Accuracy/train', acc_meter.avg, epoch * len(train_loader_stage2) + n_iter)
                # writer.add_images('images', result, epoch * len(train_loader_stage2) + n_iter)
        scheduler.step()


        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, *_) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else: 
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else: 
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                cmc, mAP, *_ = do_inference(cfg, model, val_loader, num_query, evaluator, epoch)

                if mAP + cmc[0] > best_map + best_rank1:
                    early_stop_count = 0
                    best_map = mAP
                    best_rank1 = cmc[0]
                    torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                else:
                    # pass
                    early_stop_count += 1
                    if early_stop_count > 10:
                        break
                torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query, evaluator, epoch=1, visual=False):
    device = "cuda"
    model.to(device)

    model.eval()
    all_feat = []
    all_pids = []
    # show_heatmap(model)

    # target_file = "./visual/dismat_{}.mat".format(cfg.DATASETS.NAMES)
    # if not os.path.exists(target_file):
    for n_iter, (img, pids, camids, clothes_ids, cloth, _cloth) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            feat = model(img)

            # feat = feat[:, :768]
            # list_feat = [row for row in feat]
            # all_feat.extend(list_feat)
            # list_pids = [pid for pid in pids]
            # all_pids.extend(list_pids)
            evaluator.update(output=(feat, pids, camids, clothes_ids))
    cmc, mAP, dismat, *_ = evaluator.compute(epoch)
    #     dismat_file = {
    #         "qfeat": all_feat[:num_query],
    #         "gfeat": all_feat[num_query:],
    #         "qlabel": all_pids[:num_query],
    #         "glabel": all_pids[num_query:]
    #     }
    #     scipy.io.savemat(target_file, dismat_file)
    # else:
    #     dismat = scipy.io.loadmat(target_file)
    #     qfeat = dismat["qfeat"]
    #     gfeat = dismat["gfeat"]
    #     qlabel = dismat["qlabel"]
    #     glabel = dismat["glabel"]

    if visual:
        dataset = val_loader.dataset.dataset
        # qfeat = all_feat[:num_query]
        # gfeat = all_feat[num_query:]
        # qlabel = all_pids[:num_query]
        # glabel = all_pids[num_query:]
        # query_set = dataset[:num_query]
        # gallery_set = dataset[num_query:]
        # q_paths = [data for data, _, _, _ in query_set]
        # g_paths = [data for data, _, _, _ in gallery_set]
        # visualize_ranked_results_inference(dismat,q_paths,cfg.DATASETS.NAMES,True, g_paths)

        # distance_plot(qfeat, gfeat, qlabel, glabel,cfg.DATASETS.NAMES)
        # visualize_tsne(torch.stack(all_feat), torch.stack(all_pids), cfg.DATASETS.NAMES)
    return cmc, mAP, dismat
