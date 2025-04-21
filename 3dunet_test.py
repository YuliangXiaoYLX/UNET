"""
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2023-12-14 18:10:58
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2025-04-19 23:51:41
FilePath: /Downloads/UNET/3dunet_test.py
Description:
I Love IU
Copyright (c) 2023 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import monai
import torch
import glob
from monai.networks.nets import UNet
import numpy as np
from omegaconf import OmegaConf
from utils import make_if_dont_exist
import argparse
import os
import resource
from tqdm import tqdm
import json
from monai.data import DataLoader, Dataset, write_nifti
from metrics import (
    dice_score,
    average_surface_distance,
    hausdorff_distance,
    surface_dice,
    average_normal_error,
    average_normalized_lap_distance,
)
from pytorch3d.ops.marching_cubes import marching_cubes
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes

labels = {
    "0": "background",
    "1": "myocardium",
    "2": "LA",
    "3": "LV",
    "4": "RA",
    "5": "RV",
    "6": "aorta",
    "7": "pulmonary",
}


def parse_command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None, type=str, help="path to config file")
    parser.add_argument(
        "--device", default="cuda", type=str, help="device to use for inference"
    )
    args = parser.parse_args()
    return args


def dataset(cfg, test_dir):
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    test_data = []
    for i in sorted(glob.glob(os.path.join(test_dir, "images", "*.nii.gz"))):
        test_data.append({"img": i, "seg": i.replace("images", "labels")})
    test = test_data
    transform = monai.transforms.Compose(
        transforms=[
            monai.transforms.LoadImageD(keys=["img", "seg"], image_only=False),
            monai.transforms.TransposeD(keys=["img", "seg"], indices=(2, 1, 0)),
            monai.transforms.EnsureChannelFirstD(keys=["img", "seg"]),
        ]
    )

    test_dataset = Dataset(data=test, transform=transform)
    return DataLoader(
        test_dataset, batch_size=cfg.test_bs, num_workers=cfg.num_workers, shuffle=False
    )


if __name__ == "__main__":
    args = parse_command()
    cfg = args.cfg
    device = torch.device(args.device)
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
    test_dir = os.path.join(dataset_dir, "test")
    exp_path = os.path.join(root_dir, exp)
    test_path = os.path.join(exp_path, "inference")
    model_path = os.path.join(exp_path, "model")
    make_if_dont_exist(test_path)

    test_loader = dataset(cfg, test_dir)

    # load model
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=cfg.model.class_num,
        channels=cfg.model.channels,
        strides=cfg.model.strides,
    ).to(device)
    best_model = torch.load(os.path.join(model_path, "model.pth"), map_location=device)
    model.load_state_dict(best_model["weights"])

    model.eval()
    results = {}
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="inference", unit="batch")):
            results[str(i)] = {}
            img = batch["img"].to(device)
            seg = batch["seg"].to(device)
            seg = monai.networks.one_hot(seg, cfg.model.class_num)
            pred = model(img)
            output = torch.argmax(pred.softmax(dim=1), dim=1, keepdim=True)
            output_tensor = output.permute(0, 1, 4, 3, 2).contiguous()
            output_np = output_tensor.squeeze().cpu().numpy().astype(np.uint8)
            meta = batch["img_meta_dict"]
            filename = os.path.basename(meta["filename_or_obj"][0])
            save_path = os.path.join(test_path, filename)
            # Save with original metadata
            write_nifti(
                data=output_np,  # shape: [H, W, D] or [1, H, W, D]
                file_name=save_path,
                affine=meta["affine"][0].cpu().numpy(),  # shape [4, 4]
                dtype=np.uint8,
            )
            onehot_pred = monai.networks.one_hot(output, cfg.model.class_num)
            for c in range(1, cfg.model.class_num):  # skip background
                class_name = labels[str(c)]
                results[str(i)][class_name] = {}
                onehot_pred_c = onehot_pred[:, c, ...].unsqueeze(1)
                seg_c = seg[:, c, ...].unsqueeze(1)
                pred_verts, pred_faces = marching_cubes(onehot_pred_c[:, 0, ...].float())
                gt_verts, gt_faces = marching_cubes(seg_c[:, 0, ...].float())
                pred_mesh = Meshes(verts=pred_verts, faces=pred_faces)
                gt_mesh = Meshes(verts=gt_verts, faces=gt_faces)
                spacing = batch["seg_meta_dict"]["pixdim"][0, 1:4]
                results[str(i)][class_name]["dsc"] = (
                    dice_score(onehot_pred_c, seg_c).detach().cpu().numpy().item()
                )
                results[str(i)][class_name]["asd"] = (
                    average_surface_distance(onehot_pred_c, seg_c)
                    .detach()
                    .cpu()
                    .numpy()
                    .item()
                )
                results[str(i)][class_name]["hd"] = (
                    hausdorff_distance(onehot_pred_c, seg_c)
                    .detach()
                    .cpu()
                    .numpy()
                    .item()
                )
                results[str(i)][class_name]["sd"] = surface_dice(
                    onehot_pred_c[:, 0, ...], seg_c[:, 0, ...], spacing
                )
                results[str(i)][class_name]["cd"] = chamfer_distance(
                    gt_verts[0].unsqueeze(0), pred_verts[0].unsqueeze(0)
                )[0].item()
                if pred_verts[0].shape[0] > 60000:
                    results[str(i)][class_name]["ane"] = 20.0
                else:
                    results[str(i)][class_name]["ane"] = average_normal_error(
                        gt_mesh, pred_mesh
                    ).item()
                results[str(i)][class_name]["anld"] = average_normalized_lap_distance(
                    pred_mesh
                ).item()

    # --- NEW METRICS SECTION ---
    # Initialize containers for per-class metrics
    class_names = list(labels.values())[1:]  # exclude background
    class_metrics = {name: {
        "dsc": [],
        "asd": [],
        "hd": [],
        "sd": [],
        "cd": [],
        "ane": [],
        "anld": [],
    } for name in class_names}

    # Populate metrics for each class
    for sample_result in results.values():
        for class_name in class_names:
            if class_name in sample_result:
                for metric in class_metrics[class_name]:
                    value = sample_result[class_name][metric]
                    if value is not None and not np.isnan(value):
                        class_metrics[class_name][metric].append(value)

    # Compute class-wise means
    mean_metrics = {}
    for class_name in class_names:
        mean_metrics[class_name] = {
            f"mean_{metric}": float(np.mean(values)) if values else None
            for metric, values in class_metrics[class_name].items()
        }

    # Save to JSON
    with open(os.path.join(test_path, "results.json"), "w") as f:
        json.dump(mean_metrics, f, indent=4, sort_keys=False)