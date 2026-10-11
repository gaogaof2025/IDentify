import os
import re
import glob
import logging
import numpy as np
import os.path as osp
from PIL import Image
from torch.utils import data

from datasets.ltcc import path_sep


class CelebReIDLight(object):
    """Celeb-reID Dataset Loader

    目录结构:
    - root/Celeb-reID/
      - train/
      - query/
      - gallery/
    """
    dataset_dir = "Celeb-reID-light"

    def __init__(self, root='data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self._check_before_run()


        pid2label, clothes2label, pid2clothes = self.get_pid2label_and_clothes2label(self.train_dir)


        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = self._process_dir(self.train_dir, pid2label, clothes2label , relabel=True)
        query, num_query_pids, num_query_imgs, num_query_clothes, _ = self._process_dir(self.query_dir, pid2label, clothes2label ,relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs, num_gallery_clothes, _  = self._process_dir(self.gallery_dir, pid2label, clothes2label ,relabel=False)

        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        logger = logging.getLogger('transreid.train')
        logger.info("=> Celeb-reID loaded")
        logger.info("Dataset statistics:")
        logger.info("  ------------------------------")
        logger.info("  subset   | # ids | # images")
        logger.info("  ------------------------------")
        logger.info("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        logger.info("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        logger.info("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        logger.info("  ------------------------------")
        logger.info("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        logger.info("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = len(clothes2label)
        self.pid2clothes = pid2clothes

    def _check_before_run(self):
        """检查所有文件夹是否存在"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' 不存在".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' 不存在".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' 不存在".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' 不存在".format(self.gallery_dir))

    def _process_dir(self, dir_path,pid2label, clothes2label ,relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.sort()



        pattern = re.compile(r'([-\d]+)_(\d)')

        pid_container = set()
        clothes_container = set()
        for img_path in img_paths:
            fname = osp.basename(img_path)
            pid, _ = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # 忽略垃圾图像
            pid_container.add(pid)

        pid_container = sorted(pid_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        pid2clothes = np.zeros((len(pid2label),len(clothes2label)))

        dataset = []
        for img_path in img_paths:
            fname = osp.basename(img_path)
            pid, camid = map(int, pattern.search(fname).groups())



            if pid == -1: continue  # 忽略垃圾图像
            camid -= 1  # 索引从0开始
            if relabel:
                clothes_id = camid+pid

                pid = pid2label[pid]
                pid2clothes[pid, clothes_id] = 1
            else:
                pid = pid
                clothes_id = camid


            cloth_path = img_path.replace(path_sep("train"), path_sep('train_cloth'))
            if relabel:
                dataset.append((img_path, pid, camid,clothes_id, cloth_path, cloth_path))
            else:
                dataset.append((img_path, pid, camid, clothes_id))


        return dataset, len(pid_container), len(dataset), len(clothes_container), pid2clothes

    def get_pid2label_and_clothes2label(self, dir_path):
        """获取pid到label的映射和衣服到label的映射"""
        # For example "x_y_z.jpg", "x" represents ID, "y" represents y-th image, "z" is meaningless...
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.sort()

        pid_container = set()
        clothes_container = set()
        for img_path in img_paths:
            fname = osp.basename(img_path)
            pid, camid = map(int, fname.split('_')[:2])
            if pid == -1: continue
            pid_container.add(int(pid))
            clothes_container.add(camid+pid)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        pid2clothes = np.zeros((num_pids, num_clothes))

        for img_path in img_paths:
            fname = osp.basename(img_path)
            pid, camid = map(int, fname.split('_')[:2])
            if pid == -1: continue

            pid2clothes[pid2label[pid], clothes2label[camid+pid]] = 1
        return pid2label, clothes2label, pid2clothes


