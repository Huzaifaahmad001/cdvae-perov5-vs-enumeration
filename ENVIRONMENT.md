# Software environment

The notebook builds this environment automatically in Cell 4 (about 25 minutes the first time). It uses old versions on purpose: CDVAE was written in 2021 and newer versions of its helper packages removed functions it needs.

| Package | Version | Why this version |
|---|---|---|
| Python | 3.8 | CDVAE was written for it |
| PyTorch | 1.9.0 (CUDA 10.2) | The version CDVAE was tested with. Works on a Colab T4. Newer GPUs (A100, L4, H100) are not supported by this build |
| PyTorch Lightning | 1.3.8 | Training framework used by CDVAE |
| torchmetrics | < 0.7 | Lightning 1.3.8 imports a function that newer versions deleted |
| mkl | < 2024.1 | Newer versions cause the error `undefined symbol: iJIT_NotifyEvent` |
| setuptools | <= 59.5.0 | Newer versions cause `module 'distutils' has no attribute 'version'` |
| pymatgen | 2020.12.31 | Reads CIF crystal files |
| matminer | 0.7.3 | Used by CDVAE's evaluation code |
| monty | < 2022 | pymatgen 2020 needs functions that newer monty versions removed |
| ruamel.yaml | < 0.18 | pymatgen 2020 uses a function that was removed in 0.18 |
| numpy | < 1.24 | Older code uses names removed in numpy 1.24 |
| torch-geometric | 1.7.2 | Graph neural network library |
| torch-scatter / torch-sparse / torch-cluster / torch-spline-conv | 2.0.9 / 0.6.12 / 1.5.9 / 1.2.1 | Ready-made wheels exist only for these versions with PyTorch 1.9.0 + CUDA 10.2 |
| hydra-core | 1.1.0 | CDVAE's settings system |
| wandb | 0.10.33 | Logging (run in offline mode here) |
| protobuf | < 4 | Needed by the old wandb |
| smact | 2.2.1 | Chemistry rules used by CDVAE |

Two helper changes made by the notebook:

1. **Data cache.** Cell 3 patches `cdvae/pl_data/dataset.py` so the slow data conversion (about 15 minutes per data set) is saved in Google Drive and reused.
2. **Environment backup.** Cell 4 saves the finished environment to `cdvae_env_backup.tar.gz` on Google Drive and restores it after a Colab reset.
