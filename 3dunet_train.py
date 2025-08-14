"""
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2023-11-28 13:49:29
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2025-04-20 23:41:14
FilePath: /Downloads/UNET/3dunet_train.py
Description:
I Love IU
Copyright (c) 2023 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import monai
import glob
import torch
from monai.networks.nets import UNet
from monai.transforms import Compose, Activations, AsDiscrete
import numpy as np
from omegaconf import OmegaConf
import datetime
from utils import (
    make_if_dont_exist,
    setup_logger,
    save_checkpoint,
    plot_progress,
    TqdmToLogger,
    batch_per_class_dice,
)
import argparse
import os
import logging
import resource
from tqdm import tqdm
from monai.data import DataLoader, Dataset
import pickle


def parse_command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None, type=str, help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="use this if you want to continue a training",
    )
    args = parser.parse_args()
    return args


def dataset(cfg, train_dir, test_dir):
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    train_data = []
    test_data = []
    for i in sorted(glob.glob(os.path.join(train_dir, "images", "*.nii.gz"))):
        train_data.append({"img": i, "seg": i.replace("images", "labels")})
    train, valid = monai.data.utils.partition_dataset(train_data, ratios=(7, 3))
    for i in sorted(glob.glob(os.path.join(test_dir, "images", "*.nii.gz"))):
        test_data.append({"img": i, "seg": i.replace("images", "labels")})
    test = test_data

    transform = monai.transforms.Compose(
        transforms=[
            monai.transforms.LoadImageD(keys=["img", "seg"]),
            monai.transforms.TransposeD(keys=["img", "seg"], indices=(2, 1, 0)),
            monai.transforms.EnsureChannelFirstD(keys=["img", "seg"]),
        ]
    )
    train_dataset = Dataset(data=train, transform=transform)
    if valid is not None:
        valid_dataset = Dataset(data=valid, transform=transform)
    else:
        valid_dataset = None

    test_dataset = Dataset(data=test, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_bs,
        num_workers=cfg.num_workers,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.val_bs, num_workers=cfg.num_workers, shuffle=False
    )

    test_loader = DataLoader(
        test_dataset, batch_size=cfg.test_bs, num_workers=cfg.num_workers, shuffle=False
    )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    args = parse_command()
    cfg = args.cfg
    resume = args.resume
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg is not None:
        if os.path.exists(cfg):
            cfg = OmegaConf.load(cfg)
        else:
            raise FileNotFoundError(f"config file {cfg} not found")
    else:
        raise ValueError("config file not specified")

    # setup folders
    exp = cfg.experiment
    root_dir = cfg.dataset.root_dir
    dataset_dir = os.path.join(root_dir, "dataset", "3D")
    exp_path = os.path.join(root_dir, exp)
    log_path = os.path.join(exp_path, "log")
    ckpt_path = os.path.join(exp_path, "checkpoint")
    plot_path = os.path.join(exp_path, "plot")
    test_path = os.path.join(exp_path, "inference")
    model_path = os.path.join(exp_path, "model")

    if not resume:
        make_if_dont_exist(exp_path, overwrite=True)
        make_if_dont_exist(model_path, overwrite=True)
        make_if_dont_exist(log_path, overwrite=True)
        make_if_dont_exist(ckpt_path, overwrite=True)
        make_if_dont_exist(plot_path, overwrite=True)

    datetime_object = (
        "training_log_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".log"
    )
    logger = setup_logger("EndoSAM", os.path.join(log_path, datetime_object))
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    logger.info(f"Welcome To {exp}")

    # load dataset
    logger.info("Load Dataset-Specific Parameters")
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    tr_loader, va_loader, te_loader = dataset(cfg, train_dir, test_dir)
    with open(os.path.join(exp_path, "dataset.pkl"), "wb") as f:
        pickle.dump(
            [tr_loader, va_loader, te_loader], f, protocol=pickle.HIGHEST_PROTOCOL
        )

    logger.info("Load Model-Specific Parameters")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=cfg.model.class_num,
        channels=cfg.model.channels,
        strides=cfg.model.strides,
    ).to(device)
    lr = cfg.opt_params.lr_default
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    scores = []
    best_val_score = -np.inf
    max_iter = cfg.max_iter
    val_iter = cfg.val_iter
    start_epoch = 0
    dice_ce_loss = monai.losses.DiceCELoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        reduction="mean",
        lambda_dice=cfg.losses.dice.weight,
        lambda_ce=cfg.losses.ce.weight,
    )
    post_pred = Compose(
        [AsDiscrete(argmax=True, to_onehot=cfg.model.class_num, dim=1)]
    )
    post_label = Compose([AsDiscrete(to_onehot=cfg.model.class_num, dim=1)])
    dsc = monai.metrics.DiceMetric(
        include_background=False, reduction="none", get_not_nans=True
    )

    if resume:
        ckpt = torch.load(os.path.join(ckpt_path, "ckpt.pth"), map_location=device)
        optimizer.load_state_dict(ckpt["optimizer"])
        model.load_state_dict(ckpt["weights"])
        best_val_score = ckpt["best_val_score"]
        train_losses = ckpt["train_losses"]
        scores = ckpt["scores"]
        val_losses = ckpt["val_losses"]
        lr = optimizer.param_groups[0]["lr"]
        start_epoch = ckpt["epoch"] + 1
        logger.info("Resume Training")
    else:
        logger.info("Start Training")

    for epoch in range(start_epoch, max_iter):
        train_loss = []
        model.train()
        with tqdm(tr_loader, file=tqdm_out, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{cfg.max_iter} Training")
                optimizer.zero_grad()
                img = batch["img"].to(device)
                seg = batch["seg"].to(device)
                pred = model(img)
                loss = dice_ce_loss(pred, seg)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                tepoch.set_postfix(loss=loss.item())

        tr_loss = np.mean(train_loss, axis=0)
        train_losses.append([epoch + 1, tr_loss])

        if epoch % val_iter == 0:
            model.eval()
            dsc.reset()
            valid_loss = []
            with torch.inference_mode():
                with tqdm(va_loader, file=tqdm_out, unit="batch") as tepoch:
                    for batch in tepoch:
                        tepoch.set_description(
                            f"Epoch {epoch + 1}/{cfg.max_iter} Validation"
                        )
                        img = batch["img"].to(device)
                        seg = batch["seg"].to(device).long()
                        logits = model(img)  # [B, C, ...], raw scores
                        loss = dice_ce_loss(logits, seg)
                        valid_loss.append(loss.item())

                        # post-process to one-hot for metric
                        y_pred = post_pred(logits)  # [B, C, ...] one-hot (float)
                        y_true = post_label(seg)  # [B, C, ...] one-hot (float)

                        # compute metric for current iteration
                        # accumulate metric (do NOT cast to long; keep float)
                        dsc(y_pred=y_pred, y=y_true)
                        # per-batch per-class dice
                        dice_bc = batch_per_class_dice(y_pred, y_true)  # [B, C]
                        dice_bc = dice_bc[:, 1:]  # exclude background -> [B, C-1]
                        # show class-wise mean over samples in this batch
                        per_class_now = dice_bc.mean(dim=0)  # [C-1]
                        dice_dict = {
                            f"class_{i + 1}": f"{per_class_now[i].item():.4f}"
                            for i in range(per_class_now.shape[0])
                        }
                        tepoch.set_postfix(**dice_dict)

            # aggregate the final mean dice result
            class_dice, not_nans = dsc.aggregate()
            per_class_epoch = class_dice.sum(dim=0) / not_nans.sum(dim=0).clamp(
                min=1
            )  # [C-1]
            macro_mean = per_class_epoch.mean().item()
            worst_class = per_class_epoch.min().item()
            print(macro_mean)
            scores.append([epoch + 1, macro_mean])
            val_loss = np.mean(valid_loss, axis=0)
            val_losses.append([epoch + 1, val_loss])

            if macro_mean > best_val_score:
                best_val_score = macro_mean
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_val_score,
                    train_losses,
                    val_losses,
                    scores,
                    os.path.join(model_path, "model.pth"),
                )
                logger.info(
                    f"Save Best Model @ epoch {epoch + 1} | "
                    f"macro={float(macro_mean):.4f} worst={float(worst_class):.4f}"
                )
        save_checkpoint(
            model,
            optimizer,
            epoch,
            best_val_score,
            train_losses,
            val_losses,
            scores,
            os.path.join(ckpt_path, "ckpt.pth"),
        )
        plot_progress(logger, plot_path, train_losses, val_losses, scores, "metrics")
