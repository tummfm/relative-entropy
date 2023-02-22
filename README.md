# Relative Entropy Minimization

Implementation of the Relative Entropy Minimization and Force Matching methods
as employed in the paper
[Deep coarse-grained potentials via relative entropy minimization](https://aip.scitation.org/doi/10.1063/5.0124538).

## Getting started
This repository provides code to train and simulate two systems, coarse-grained
water and alanine dipeptide, with force matching and relative
entropy minimization. The training examples can be found in
[CG_water_force_matching.py](examples/water/CG_water_force_matching.py)
and [CG_water_relative_entropy.py](examples/water/CG_water_relative_entropy.py)
for water and in [alanine_force_matching.py](examples/alanine_dipeptide/alanine_force_matching.py)
and [alanine_relative_entropy.py](examples/alanine_dipeptide/alanine_relative_entropy.py)
for alanine dipeptide. Training the model with force matching will take a few
hours and more than a day with relative entropy.

MD simulation employing the trained DimeNet++ models can be found
in [CG_water_simulation.py](examples/water/CG_water_simulation.py)
and [alanine_simulation.py](examples/alanine_dipeptide/alanine_simulation.py)
respectively.

## Data sets
The data sets for alanine dipeptide and water can be downloaded from Google
Drive via the following link:</br>
[https://drive.google.com/drive/folders/1IBZbuSBIBhvFbVhuo9s-ENE2IyWG-YI_?usp=sharing](https://drive.google.com/drive/folders/1IBZbuSBIBhvFbVhuo9s-ENE2IyWG-YI_?usp=sharing)</br>
Once downloaded, you can move the conf and force files into the dataset folder
of [water](examples/water/data/dataset) and
[alanine dipeptide](examples/alanine_dipeptide/data/dataset).

## Installation
All dependencies can be installed locally with pip:
```
pip install -e .[all]
```

However, this only installs a CPU version of Jax. If you want to enable GPU 
support, please overwrite the jaxlib version:
```
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

## Requirements
The repository uses with the following packages:
```
    'jax>=0.4.3',
    'jax-md>=0.2.5',
    'optax>=0.0.9',
    'dm-haiku>=0.0.9',
    'sympy',
    'cloudpickle',
    'chex',
    'jax-sgmc',
```
The code was run with Python 3.8. The packages used in the paper 
are listed in [setup.py](setup.py).

## Citation
Please cite our paper if you use this code in your own work:
```
@article{thaler_entropy_2022,
  title = {Deep coarse-grained potentials via relative entropy minimization},
  author = {Thaler, Stephan  and Stupp, Maximilian and Zavadlav, Julija},
  journal={The Journal of Chemical Physics},
  volume = {157},
  number = {24},
  pages = {244103},
  year = {2022},
  doi = {10.1063/5.0124538}
}
```
