# A Simple for 2D and 3D U-Net Pytorch Tutorial
The results trained and tested on Liver dataset from [Medical Segmentation Decathlon](http://medicaldecathlon.com/). Checkpoints, loss plots, inference results etc are stored in the ```3D UNet Liver``` and ```2D UNet Liver```

# Usage

- Clone the repository

  ```bash
  git clone https://github.com/YuliangXiaoYLX/UNET.git && cd UNET
  ```

- Train (modify the `YAML` as needed)

  Simply run

  ```bash
  python 2dunet_train.py --cfg 2dunet_config.yaml
  ```
  
  or
  
  ```bash
  python 3dunet_train.py --cfg 3dunet_config.yaml
  ```

  for `2D` and `3D` U-Net, respectively. Add `--resume` if you need to continue your previous training.

- Test

  Simply run

  ```bash
  python 2dunet_test.py --cfg 2dunet_config.yaml
  ```
  
  or
  
  ```bash
  python 3dunet_test.py --cfg 3dunet_config.yaml
  ```
  
## Metrics

Multiple evaluation metrics are added to this tutorial:
- Mean Dice Score
- Mean Surface Distance
- Mean Hausdorff Distance
- Mean Surface Dice Score
- Mean Chamfer Distance

## Star History

<p align="center">
  <a href="https://www.star-history.com/#mikami520/UNET&Date">
   <picture>
     <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=mikami520/UNET&type=Date&theme=dark" />
     <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=mikami520/UNET&type=Date" />
     <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mikami520/UNET&type=Date" />
   </picture>
  </a>
</p>

---
<div align="center">
<p>Developed by Chris Xiao | University of Toronto</p>
<p>© 2025 All Rights Reserved</p>
</div>
