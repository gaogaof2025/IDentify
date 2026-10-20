from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def read_image_gray(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('L')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")




class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id, cloth_path, _cloth_path = self.dataset[index]
        img = read_image(img_path)
        cloth = read_image_gray(cloth_path)  ##################################
        _cloth = read_image_gray(_cloth_path)  ####################################
        # contour = read_image_gray(contour_path)
        # print (np.max(cloth), np.max(_cloth), np.max(contour))
        if self.transform is not None:
            img, cloth, _cloth = self.transform(img, cloth, _cloth)
        return img, pid, camid, clothes_id, cloth, _cloth


class ImageDataset_Val(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, transform_gray=None):
        self.dataset = dataset
        self.transform = transform
        self.transform_gray = transform_gray

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        img = read_image(img_path)

        if osp.exists(img_path.replace('/query/', '/query_cloth/')):
            cloth = read_image_gray(img_path.replace('/query/', '/query_cloth/'))  ########################
        else:
            cloth = read_image_gray(img_path)

        if osp.exists(img_path.replace('/query/', '/query_cloth_/')):
            _cloth = read_image_gray(img_path.replace('/query/', '/query_cloth_/'))  #######################
        else:
            _cloth = read_image_gray(img_path)

        # if osp.exists(img_path.replace('/query/', '/query_contour/')):
        #     contour = read_image_gray(img_path.replace('/query/', '/query_contour/'))
        # else:
        #     contour = read_image_gray(img_path)

        if self.transform is not None:
            img, cloth, _cloth = self.transform(img, cloth, _cloth)

            # img = self.transform(img)
            # cloth = self.transform_gray(cloth)
            # _cloth = self.transform_gray(_cloth)
            # contour = 0.0 - self.transform_gray(contour)
        return img, pid, camid, clothes_id, cloth, _cloth  # , img_path