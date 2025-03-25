# Multi-Modal MIL Models for Discrete-Time Survival Prediction

This repository contains an adaptation of the AMIL, DeepSet, DeepAttnMISL, PORPOISE, MCAT, and MOTCat models for discrete-time survival prediction tasks using whole-slide images and genetic data.

## Installation

Tested on:
- Ubuntu 22.04
- Nvidia GeForce RTX 4090
- Python 3.10
- PyTorch 2.3

Clone the repository and navigate to the directory.

```bash
git clone https://github.com/ezgiogulmus/MMSurv.git
cd MMSurv
```

Create a conda environment and install required packages.

```bash
conda create -n mmsurv python=3.10 -y
conda activate mmsurv
pip install --upgrade pip 
pip install -e .
```

## Usage

First, extract patch coordinates and patch-level features using the CLAM library available at [CLAM GitHub](https://github.com/Mahmoodlab/CLAM).

To try it on the dummy data, run the following commands in the [mmsurv](./mmsurv) directory:

```bash
python create_dummydata.py
python save_cluster_ids.py dummy --patch_dir ./dummy_data/coords_dir/
```

Then run the model:

```bash
python main.py --data_name dummy --feats_dir ./dummy_data/feats_dir/ --omics rna,dna,cnv --model_type porpoise
```

- `model_type`: Options are `'deepset'`, `'amil'`, `'deepattnmisl'`, `'mcat'`, `'motcat'`, `'porpoise'`  
See [mmsurv](./mmsurv/arguments.py) for detailed configuration options.

## Acknowledgement

This code is adapted from the repositories of:
- [PORPOISE](https://github.com/mahmoodlab/PORPOISE)
- [MCAT](https://github.com/mahmoodlab/MCAT)
- [MOTCat](https://github.com/Innse/MOTCat)

## License

This repository is licensed under the [GPLv3 License](./LICENSE). Note that this project is for non-commercial academic use only, in accordance with the licenses of the original models.

## References

Chen, Richard J., et al. "Multimodal Co-Attention Transformer for Survival Prediction in Gigapixel Whole Slide Images." *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2021, pp. 4015-4025.

Chen, Richard J., et al. "Pan-cancer integrative histology-genomic analysis via multimodal deep learning." *Cancer Cell*, 2022.

Xu, Yingxue, and Hao Chen. "Multimodal Optimal Transport-based Co-Attention Transformer with Global Structure Consistency for Survival Prediction." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, pp. 21241-21251.

Maximilian Ilse, Jakub Tomczak, and Max Welling. "Attention-based deep multiple instance learning." *In Proceedings of the 35th International Conference on Machine Learning*, pages 2132–2141, 2018.

Jiawen Yao, Xinliang Zhu, Jitendra Jonnagaddala, Nicholas Hawkins, and Junzhou Huang. "Whole slide images based cancer survival prediction using attention guided deep multiple instance learning networks." *Medical Image Analysis*, 65:101789, 2020.

Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan Salakhutdinov, and Alexander Smola. "Deep sets". *Advances in Neural Information Processing Systems*, 2017.
