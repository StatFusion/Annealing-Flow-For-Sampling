## Citation
```bibtex
@inproceedings{wu2025annealing,
  title={Annealing Flow Generative Model Towards Sampling High-Dimensional and Multi-Modal Distributions},
  author={Wu, Dongze and Xie, Yao},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year={2025}
}

## Requirements

- `torch`
- `torchdiffeq`
- `scipy`
- `sklearn`
- `Pillow`

## Usage Instructions

Use `Annealing_Flow.py` for training the samplers and obtaining samples.

### To Run `Annealing_Flow.py`:

1. Navigate to the main project directory using `cd`.

2. Then to train 50D ExpGauss with 1024 modes, for example, run:
   ```python
   python Annealing_Flow.py --AnnealingFlow_config ExpGauss.yaml

### Configuration Files:
1. GMM_sphere_c={radius}.yaml: GMMs with different numbers of means aligned on a circle with radius c.
2. truncated.yaml: Truncated Normal distribution with varying dimensions and radius c, i.e., 1_{||x||>c}*N(0,I_{d})
3. funnel.yaml: Funnel distribution on 5D space
4. ExpGauss.yaml: Exp-Weighted Gaussian distribution with 1024 modes on 50D space
5. ExpGauss_unequal.yaml: Unequally weighted Exp-Weighted Gaussian distribution with 1024 modes on 50D space

For Bayesian logistics, please run Annealing_Flow.py in a similar way from the Bayesian_Logistics folder.

### After running `Annealing_Flow.py`:

All trained velocity fields are saved in: 
`/samplers_trained/{distributions you trained}/block_{i}.pth`.

You can then follow a similar main loop as in `Annealing_Flow.py` to generate new samples using the saved neural networks.

