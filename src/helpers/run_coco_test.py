import os
import glob
import math
import numpy as np
import random
import pickle
import PIL
import torch
from torchgeometry.losses import ssim


BBOX_PATH = r"/kaggle/input/mscoco2017-face-bbox/test2017.txt"


def get_bbox():
    with open(BBOX_PATH, "r") as fin:
        bbox_dict = dict()
        lines = fin.readlines()
        for line in lines:
            f_name, *coord = line.rstrip().split(" ")
            coord = tuple(map(float, coord))
            if f_name in bbox_dict:
                bbox_dict[f_name].append(coord)
            else:
                bbox_dict[f_name] = [coord]
    return bbox_dict


def mse(original, reconstructed):
    mse = (np.square(original - reconstructed)).mean()
    return mse


def masked_ssim(original, reconstructed):
    loss = 1 - ssim(original, reconstructed, window_size=11, reduction='mean')
    return loss


def calculate_loss(origin_path, recon_path):
    recon_imgs = glob.glob(os.path.join(recon_path, '*.jpg'))
    recon_imgs += glob.glob(os.path.join(recon_path, '*.png'))

    mse_loss = []
    ssim_loss = []
    fid_loss = []

    bbox_dict = get_bbox()

    for img_path in recon_imgs:
        fname = os.path.basename(img_path)

        if fname not in bbox_dict:
            continue

        origin_path = glob.glob(origin_path + f"/*{fname}")[0]


        recon_img = PIL.Image.open(img_path)
        recon_img = recon_img.convert('RGB')
        origin_img = PIL.Image.open(origin_path)
        origin_img = origin_img.convert('RGB')

        W, H = recon_img.size  # slightly confusing

        bbox = torch.tensor(bbox_dict[fname])
        mask = torch.zeros((H, W))
        bbox[:, (0, 2)] *= W
        bbox[:, (1, 3)] *= H
        # (x1, y1) upper-left corner of face rectangle, (x2, y2) - lower-right corner
        for (x1, y1, x2, y2) in bbox:
            mask[int(x1): int(x2), int(y1): int(y2)] = 1

        recon_img = recon_img * mask
        origin_img = origin_img * mask

        mse_ = mse(origin_img, recon_img)
        ssim_ = masked_ssim(origin_img, recon_img)
        
        mse_loss.append(mse_)
        ssim_loss.append(ssim_)

    return mse_loss, ssim_loss, fid_loss


