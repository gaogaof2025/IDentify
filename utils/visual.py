import os
import re

import cv2
import numpy as np
import tqdm
from PIL import Image

cam2label = {'A': 0, 'B': 1, 'C': 2}
ltcc_cc_ids = [6,8,9,11,57,58,137,105,106,78,29,79,82,83,31,90,46,59,60,16,17,143,110,118,24,25,52,28,33,63,101,134,142,145,65,20,32,37,74,125,84,104,141,120,72]
ltcc_sc_ids = [1,12,80,139,107,108,112,114,49,55,14,62,67,26,99,126,102,19,23,121,77,119,73,68,38,147,148,113,41,42]

def visualize_ranked_results_inference(distmat, q_paths, dataset_name, show_cc,g_paths, width=128, height=256, topk=10,
                                       GRID_SPACING=10, QUERY_EXTRA_SPACING=90, BW=5):
    """Visualizes ranked results.

    Supports both image-reid and video-reid.

    For image-reid, ranks will be plotted in a single figure. For video-reid, ranks will be
    saved in folders each containing a tracklet.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    num_q, num_g = distmat.shape

    indices = np.argsort(distmat, axis=1)

    num_cols = topk + 1
    grid_img = 255 * np.ones((height, num_cols * width + (topk - 1) * GRID_SPACING + QUERY_EXTRA_SPACING, 3),
                             dtype=np.uint8)

    for q_idx, qimg_path in tqdm.tqdm(enumerate(q_paths)):
        qimg = cv2.imread(qimg_path)
        qimg = Image.fromarray(cv2.cvtColor(qimg, cv2.COLOR_BGR2RGB))
        qimg = cv2.cvtColor(np.asarray(qimg), cv2.COLOR_RGB2BGR)

        qimg = cv2.resize(qimg, (width, height))
        grid_img[:height, :width, :] = qimg

        if dataset_name == 'ltcc':
            pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
            pattern2 = re.compile(r'(\w+)_c')
            q_pid, _, q_camid = map(int, pattern1.search(qimg_path).groups())
            q_cloid = pattern2.search(qimg_path).group(1)
        elif dataset_name == 'prcc':
            q_pid = os.path.basename(os.path.dirname(qimg_path))
            q_camid = os.path.basename(os.path.dirname(os.path.dirname(qimg_path)))
            q_camid = cam2label[q_camid]
            q_cloid = q_camid

        if show_cc == True:
            if dataset_name == 'ltcc' and q_pid not in ltcc_cc_ids:
                continue

        if show_cc == False:
            if dataset_name == 'ltcc' and q_pid not in ltcc_sc_ids:
                continue

        cnt = 0
        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            gimg_path = g_paths[g_idx]

            if dataset_name == 'ltcc':

                g_pid, _, g_camid = map(int, pattern1.search(gimg_path).groups())
                g_cloid = pattern2.search(gimg_path).group(1)
            elif dataset_name == 'prcc':
                g_pid = os.path.basename(os.path.dirname(gimg_path))
                g_camid = os.path.basename(os.path.dirname(os.path.dirname(gimg_path)))
                g_camid = cam2label[g_camid]
                g_cloid = g_camid

            ## ALL show
            if q_pid == g_pid and q_camid == g_camid:
                continue

            ## CC show
            if show_cc == True:
                if q_pid == g_pid and q_cloid == g_cloid:
                    continue

            if q_pid == g_pid:
                border_color = (0, 255, 0)
            else:
                border_color = (0, 0, 255)

            gimg = cv2.imread(gimg_path)
            gimg = Image.fromarray(cv2.cvtColor(gimg, cv2.COLOR_BGR2RGB))
            gimg = cv2.cvtColor(np.asarray(gimg), cv2.COLOR_RGB2BGR)

            gimg = cv2.resize(gimg, (width, height))
            gimg = cv2.copyMakeBorder(gimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color)
            gimg = cv2.resize(gimg, (width, height))

            start = rank_idx * width + (rank_idx - 1) * GRID_SPACING + QUERY_EXTRA_SPACING
            end = (rank_idx + 1) * width + (rank_idx - 1) * GRID_SPACING + QUERY_EXTRA_SPACING
            grid_img[:height, start:end, :] = gimg

            rank_idx += 1
            if rank_idx > topk:
                break
        if cnt == 0:
            cnt += 1

        save_path = os.path.abspath(os.path.join("./visual",dataset_name,qimg_path.split(os.path.sep)[-1]))
        if not os.path.exists(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, grid_img)