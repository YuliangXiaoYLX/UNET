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
import nibabel as nib
from monai.data import DataLoader, Dataset, write_nifti
from metrics import dice_score, average_surface_distance, hausdorff_distance, surface_dice, average_normal_error, average_normalized_lap_distance
from pytorch3d.ops.marching_cubes import marching_cubes
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes

def parse_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None, type=str, help='path to config file')
    args = parser.parse_args()
    return args

def dataset(cfg, test_dir):
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    test_data = []
    for i in sorted(glob.glob(os.path.join(test_dir, 'images', '*.nii.gz'))):
        test_data.append({
            'img': i,
            'seg': i.replace('images', 'labels')
        })
    transform = monai.transforms.Compose([
        monai.transforms.LoadImageD(keys=['img', 'seg'], image_only=False),
        monai.transforms.EnsureChannelFirstD(keys=['img', 'seg']),
    ])
    test_dataset = Dataset(data=test_data, transform=transform)
    return DataLoader(test_dataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)

def restructure_results(results, num_classes=8, background_class=0):
    num_metrics = 7
    per_class_metrics = {cls: [] for cls in range(num_classes) if cls != background_class}
    for case_metrics in results.values():
        for cls in range(1, num_classes):
            cls_metrics = (
                case_metrics.get(f'dsc_{cls}', 0.0),
                case_metrics.get(f'asd_{cls}', 0.0),
                case_metrics.get(f'hd_{cls}', 0.0),
                case_metrics.get(f'sd_{cls}', 0.0),
                case_metrics.get(f'cd_{cls}', 0.0),
                case_metrics.get(f'ane_{cls}', 0.0),
                case_metrics.get(f'anld_{cls}', 0.0),
            )
            per_class_metrics[cls].append(cls_metrics)

    metrics_array = np.zeros((num_classes - 1, len(per_class_metrics[1]), num_metrics))
    for idx, cls in enumerate(range(1, num_classes)):
        metrics_array[idx] = np.array(per_class_metrics[cls])

    mean_metrics_per_class = metrics_array.mean(axis=1)
    overall_mean_metrics = mean_metrics_per_class.mean(axis=0)

    return {
        'mean_dsc': overall_mean_metrics[0],
        'mean_asd': overall_mean_metrics[1],
        'mean_hd': overall_mean_metrics[2],
        'mean_sd': overall_mean_metrics[3],
        'mean_cd': overall_mean_metrics[4],
        'mean_ane': overall_mean_metrics[5],
        'mean_anld': overall_mean_metrics[6],
    }

if __name__ == '__main__':
    args = parse_command()
    cfg = args.cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg is not None and os.path.exists(cfg):
        cfg = OmegaConf.load(cfg)
    else:
        raise ValueError('Config file not found or unspecified.')

    exp = cfg.experiment
    root_dir = cfg.dataset.root_dir
    dataset_dir = os.path.join(root_dir, "dataset", '2D')
    test_dir = os.path.join(dataset_dir, 'test')
    exp_path = os.path.join(root_dir, exp)
    test_path = os.path.join(exp_path, 'inference')
    model_path = os.path.join(exp_path, 'model')
    make_if_dont_exist(test_path)

    test_loader = dataset(cfg, test_dir)

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=cfg.model.class_num,
        channels=cfg.model.channels,
        strides=cfg.model.strides,
    ).to(device)
    best_model = torch.load(os.path.join(model_path, 'model.pth'), map_location=device)
    model.load_state_dict(best_model['weights'])

    model.eval()
    results = {}
    all_class_metrics = {cls: [] for cls in range(1, cfg.model.class_num)}
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc='inference', unit='batch')):
            results[str(i)] = {}
            img = batch['img'].to(device)
            seg = batch['seg'].to(device)
            pred = torch.zeros_like(seg)
            for j in range(img.shape[-1]):
                if len(torch.unique(seg[..., j])) > 1:
                    output = model(img[..., j])
                    output = torch.argmax(output.softmax(dim=1), dim=1, keepdim=True)
                    pred[..., j] = output
                    del output
                    torch.cuda.empty_cache()

            seg_oh = monai.networks.one_hot(seg, cfg.model.class_num)
            pred_oh = monai.networks.one_hot(pred, cfg.model.class_num)
            spacing = batch['seg_meta_dict']['pixdim'][0, 1:4]

            for cls in range(1, cfg.model.class_num):
                pred_oh_c = pred_oh[:, cls, ...].unsqueeze(1)
                seg_oh_c = seg_oh[:, cls, ...].unsqueeze(1)
                pred_verts, pred_faces = marching_cubes(pred_oh_c[:, 0, ...].float())
                gt_verts, gt_faces = marching_cubes(seg_oh_c[:, 0, ...].float())
                pred_mesh = Meshes(verts=pred_verts, faces=pred_faces)
                gt_mesh = Meshes(verts=gt_verts, faces=gt_faces)
                dsc = dice_score(pred_oh_c, seg_oh_c).item()
                asd = average_surface_distance(pred_oh_c, seg_oh_c).item()
                hd = hausdorff_distance(pred_oh_c, seg_oh_c).item()
                sd = surface_dice(pred_oh[:, 0, ...], seg_oh[:, 0, ...], spacing)
                cd = chamfer_distance(gt_verts[0].unsqueeze(0), pred_verts[0].unsqueeze(0))[0].item()
                if pred_verts[0].shape[0] > 60000:
                    ane = 20.0
                else:
                    ane = average_normal_error(gt_mesh, pred_mesh).item()
                anld = average_normalized_lap_distance(
                    pred_mesh
                ).item()
                
                results[str(i)][f'class_{cls}'] = {
                    'dsc': dsc,
                    'asd': asd,
                    'hd': hd,
                    'sd': sd,
                    'cd': cd,
                    'ane': ane,
                    'anld': anld,
                }

                all_class_metrics[cls].append([dsc, asd, hd, sd, cd, ane, anld])

            # Save 3D prediction as nifti
            pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)
            meta = batch["img_meta_dict"]
            filename = os.path.basename(meta["filename_or_obj"][0])
            save_path = os.path.join(test_path, filename)
            # Save with original metadata
            write_nifti(
                data=pred_np,  # shape: [H, W, D] or [1, H, W, D]
                file_name=save_path,
                affine=meta["affine"][0].cpu().numpy(),  # shape [4, 4]
                dtype=np.uint8,
            )

    mean_metrics = {}
    for cls, metrics in all_class_metrics.items():
        metrics_np = np.array(metrics)
        mean_metrics[f'class_{cls}'] = {
            'mean_dsc': float(np.mean(metrics_np[:, 0])),
            'mean_asd': float(np.mean(metrics_np[:, 1])),
            'mean_hd': float(np.mean(metrics_np[:, 2])),
            'mean_sd': float(np.mean(metrics_np[:, 3])),
            'mean_cd': float(np.mean(metrics_np[:, 4])),
            'mean_ane': float(np.mean(metrics_np[:, 5])),
            'mean_anld': float(np.mean(metrics_np[:, 6])),
        }

    final_output = {
        'per_case_metrics': results,
        'mean_metrics_per_class': mean_metrics
    }

    with open(os.path.join(test_path, 'results.json'), 'w') as f:
        json.dump(final_output, f, indent=4)

    torch.cuda.empty_cache()
