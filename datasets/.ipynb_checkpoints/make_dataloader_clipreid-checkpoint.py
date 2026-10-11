import torch
import datasets.img_transforms as T

from torch.utils.data import DataLoader

from .bases import ImageDataset, ImageDataset_Val
from timm.data.random_erasing import RandomErasing

from .last import LaST
from .sampler import RandomIdentitySampler

from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .ltcc import LTCC
from .prcc import PRCC
from .ccvid import CCVID
from .celebreID import CelebReID
from .celebreID_light import CelebReIDLight
from .deepchange import DeepChange

__factory = {
    'ltcc': LTCC,
    'prcc': PRCC,
    'CCVID': CCVID,
    'last': LaST,
    'Celebreid': CelebReID,
    'Celebreid_light': CelebReIDLight,
    'deepchange': DeepChange,
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camid, clothes_id, cloth, _cloth = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camid, dtype=torch.int64)
    clothes_id = torch.tensor(clothes_id, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, clothes_id, cloth, _cloth

def val_collate_fn(batch):
    imgs, pids, camid, clothes_id, cloth, _cloth = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camid, dtype=torch.int64)
    clothes_id = torch.tensor(clothes_id, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, clothes_id, cloth, _cloth


def build_img_transforms(config):
    transform_train = T.Compose([
        T.Resize((config.INPUT.SIZE_TRAIN[0], config.INPUT.SIZE_TRAIN[1])),
        T.RandomCroping(p=config.INPUT.PROB),
        T.RandomHorizontalFlip(p=config.INPUT.RF_PROB),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=config.INPUT.RE_PROB)
    ])
    transform_test = T.Compose([
        T.Resize((config.INPUT.SIZE_TEST[0], config.INPUT.SIZE_TEST[1])),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms_gray = T.Compose([
        T.Resize(config.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=config.INPUT.PIXEL_MEAN, std=config.INPUT.PIXEL_STD)
    ])

    return transform_train, transform_test, val_transforms_gray

def make_dataloader(cfg):
    train_transforms, val_transforms, val_transforms_gray = build_img_transforms(cfg)


    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    
    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    # cam_num = dataset.num_train_cams
    # view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.STAGE2.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader_stage2 = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader_stage2 = DataLoader(
                train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader_stage2 = DataLoader(
            train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    # val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    if cfg.DATASETS.NAMES == 'prcc':
        val_set = ImageDataset_Val(dataset.query_same + dataset.query_diff + dataset.gallery, val_transforms, val_transforms_gray)
        num_query_same = len(dataset.query_same)
        num_query_diff = len(dataset.query_diff)
    else:
        val_set = ImageDataset_Val(dataset.query + dataset.gallery, val_transforms, val_transforms_gray)
        num_query_same = None
        num_query_diff = None

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_stage1 = DataLoader(
        train_set_normal, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn
    )

    pid2clothes = torch.from_numpy(dataset.pid2clothes)

    len_val = len(dataset.query_same + dataset.query_diff) if cfg.DATASETS.NAMES == 'prcc' else len(dataset.query)



    return train_loader_stage2, train_loader_stage1, val_loader, len_val, num_classes, pid2clothes, dataset.num_train_clothes, num_query_same, num_query_diff
