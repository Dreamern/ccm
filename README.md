# 🖌️ CVPR 2025 | Curriculum Consistency Model

[![arXiv](https://img.shields.io/badge/arXiv-2412.06295-b31b1b.svg)](https://arxiv.org/abs/2412.06295)

This repository contains the implementation of the CVPR2025 paper "[See Further When Clear: Curriculum Consistency Model](https://arxiv.org/abs/2412.06295)".


## 🚀 News

**Aug 5, 2025**:fire:

- We have open-sourced the model weights for CIFAR-10 (FID=1.64).

## Get Started

**Recommend Environment:** `cuda 12.1` + `python 3.9`

```bash
# Clone the Repository
git clone git@github.com:Dreamern/ccm.git

# Create Virtual Environment with Conda
conda create --name ccm python=3.9
conda activate ccm

# Install Dependencies
pip install -r requirements.txt
```

## Evaluation

Place your model weights into the `ckpts` directory. We have already provided the weights for [cifar10](https://drive.google.com/file/d/1bG7VIx_hU-GOm3esQ_xhztqv7k0VpE0R/view?usp=drive_link).

Download the [inception-2015-12-05.pt](https://drive.google.com/file/d/1fDu24oXm3Xl_KEXzToIpNv2Mv28vrLAe/view?usp=drive_link) file to the `/tmp` directory.

Download [cifar10_legacy_tensorflow_train_32.npz](https://drive.google.com/file/d/14OE52Ek-EqbdgI93z4gICGVd8BBP3P4N/view?usp=drive_link) into the `envs/ccm/lib/python3.9/site-packages/cleanfid/stats/`  directory.

Then you can compute the fid for CCM by running the following command:

```bash
bash eval.sh ckpts/otcfm_cifar10.pt
```

## Contact Us

**Yunpeng Liu**: lypniuyou@163.com

## BibTeX

```
@inproceedings{liu2025see,
  title={See Further When Clear: Curriculum Consistency Model},
  author={Liu, Yunpeng and Liu, Boxiao and Zhang, Yi and Hou, Xingzhong and Song, Guanglu and Liu, Yu and You, Haihang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={18103--18112},
  year={2025}
}
```

## Thanks
A Large portion of this codebase is built upon [torchcfm](https://github.com/atong01/conditional-flow-matching/tree/main).