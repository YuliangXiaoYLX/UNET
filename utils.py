'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2023-09-16 19:47:31
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2023-11-28 20:22:53
FilePath: /UNET/utils.py
Description: EndoSAM utilities functions 
I Love IU
Copyright (c) 2023 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''

import os
import numpy as np
import shutil
import logging
from torch.nn import functional as F
import torch
import torch.nn as nn
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple
import io


# helper: per-batch, per-class Dice (works for 2D/3D)
def batch_per_class_dice(y_pred_oh, y_true_oh, eps=1e-8):
    """
    y_pred_oh, y_true_oh: one-hot float tensors [B, C, ...]
    returns: dice [B, C]
    """
    assert y_pred_oh.shape == y_true_oh.shape
    dims = tuple(range(2, y_pred_oh.ndim))  # sum over spatial dims
    inter = (y_pred_oh * y_true_oh).sum(dim=dims)                 # [B, C]
    card  = y_pred_oh.sum(dim=dims) + y_true_oh.sum(dim=dims)     # [B, C]
    dice  = (2.0 * inter + eps) / (card + eps)                     # [B, C]
    return dice

def plot_progress(logger, save_dir, train_loss, val_loss, scores, name):
    """
    Should probably by improved
    :return:
    """
    assert len(train_loss) != 0
    train_loss = np.array(train_loss)
    try:
        font = {'weight': 'normal',
                'size': 18}

        matplotlib.rc('font', **font)

        fig = plt.figure(figsize=(30, 24))
        ax = fig.add_subplot(111)
        ax1 = ax.twinx()
        ax.plot(train_loss[:,0], train_loss[:,1], color='b', ls='-', label="loss_tr")
        if len(val_loss) != 0:
            scores = np.array(scores)
            val_loss = np.array(val_loss)
            ax.plot(val_loss[:, 0], val_loss[:, 1], color='r', ls='-', label="loss_val")
            ax.plot(scores[:, 0], scores[:, 1], color='g', ls='-', label="dsc_val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax1.set_ylabel("dice score")
        ax.legend()
        ax.set_title(name)
        fig.savefig(os.path.join(save_dir, name + ".png"))
        plt.cla()
        plt.close(fig)
    except:
        logger.info(f"failed to plot {name} training progress")


def save_checkpoint(adapter_model, optimizer, epoch, best_val_score, train_losses, val_losses, scores, save_dir):
    torch.save({
                'epoch': epoch,
                'best_val_score': best_val_score,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'weights': adapter_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scores': scores
            }, save_dir)


def one_hot_embedding_3d(labels, dim=1, class_num=21):
    '''
    :param real_labels: B 1 H W
    :param class_num: N
    :return: B N H W
    '''
    one_hot_labels = labels.clone()
    data_dim = list(one_hot_labels.shape)
    if data_dim[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")
    data_dim[dim] = class_num
    o = torch.zeros(size=data_dim, dtype=one_hot_labels.dtype, device=one_hot_labels.device)
    return o.scatter_(dim, one_hot_labels, 1).contiguous().float()


def setup_logger(logger_name, log_file, level=logging.INFO):
    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    log_setup.setLevel(level)
    log_setup.propagate = False
    if not log_setup.handlers:
        fileHandler = logging.FileHandler(log_file, mode='w')
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        fileHandler.setLevel(level)
        streamHandler.setLevel(level)
        log_setup.addHandler(fileHandler)
        log_setup.addHandler(streamHandler)
    
    return log_setup


def make_if_dont_exist(folder_path, overwrite=False):
    if os.path.exists(folder_path):
        if not overwrite:
            print(f'{folder_path} exists, no overwrite here.')
        else:
            print(f"{folder_path} overwritten")
            shutil.rmtree(folder_path, ignore_errors = True)
            os.makedirs(folder_path)
    else:
        os.makedirs(folder_path)
        print(f"{folder_path} created!")


# taken from sam.postprocess_masks of https://github.com/facebookresearch/segment-anything
def postprocess_masks(masks, input_size, original_size):
    """
    Remove padding and upscale masks to the original image size.

    Arguments:
        masks (torch.Tensor): Batched masks from the mask_decoder,
        in BxCxHxW format.
        input_size (tuple(int, int)): The size of the image input to the
        model, in (H, W) format. Used to remove padding.
        original_size (tuple(int, int)): The original size of the image
        before resizing for input to the model, in (H, W) format.

    Returns:
        (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
        is given by original_size.
    """
    masks = F.interpolate(
        masks,
        (1024, 1024),
        mode="bilinear",
        align_corners=False,
    )
    masks = masks[..., : input_size[0], : input_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks


def preprocess(x: torch.Tensor, img_size: int) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        pixel_mean=[123.675, 116.28, 103.53]
        pixel_std=[58.395, 57.12, 57.375]
        pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        x = (x - pixel_mean) / pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


class ResizeLongestSide:
    """
    Resizes images to longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)