## Requirements

- `torch`
- `torchdiffeq`
- `scipy`
- `sklearn`
- `Pillow`

## Usage Instructions

### Training the Samplers

Use `Annealing_Flow.py` for training the samplers. After the samplers are trained, use `Annealing_Flow_sampler.py` for sampling.

#### To Run `Annealing_Flow.py`:

1. Navigate to the main project directory using `cd`.

2. Then to train a GMM with 6 modes on a sphere, for example, run:
   ```python
   python Annealing_Flow.py --AnnealingFlow_config GMM_sphere_6_means.yaml

### Configuration Files:
1. GMM_sphere_{number}_means.yaml: GMMs with different numbers of means aligned on a circle
2. truncated.yaml: Truncated Normal distribution with varying dimensions and radius c, i.e., 1_{||x||>c}*N(0,I_{d})
3. funnel.yaml: Funnel distribution on 5D space
4. exponential.yaml: Exp-Weighted Gaussian distribution with 1024 modes on 10D space

For Bayesian logistics, please run Annealing_Flow.py in a similar way from the Bayesian_Logistics folder.

#### To Run `Annealing_Flow_sampler.py`:

1. Make sure you have run Annealing_Flow.py already
  
2. Update the `master_dir` to the correct master directory inside the block:
   ```python
   if __name__ == '__main__':
   
3. Modify the distributions you want to sample inside Samplers.yaml.

4. Run the following command:
   ```python
   python Annealing_Flow_sampler.py --AnnealingFlow_config Samplers.yaml

