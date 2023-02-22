"""Training a CG model for alanine dipeptide via force matching."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')  # disable warnings about float64 usage

import cloudpickle as pickle
from pathlib import Path

from jax import random
from jax_md import space
import matplotlib.pyplot as plt
import optax

from chemtrain import trainers, data_processing
from chemtrain.jax_md_mod import custom_space, io
from util import Initialization

Path('output/figures').mkdir(parents=True, exist_ok=True)
Path('output/force_matching').mkdir(parents=True, exist_ok=True)

# user input
configuration_str = 'data/dataset/confs_heavy_100ns.npy'
force_str = 'data/dataset/forces_heavy_100ns.npy'
file_topology = 'data/confs/heavy_2_7nm.gro'

training_name = 'FM_model_alanine'
save_plot = f'output/figures/FM_losses_alanine_{training_name}.png'
save_params_path = ('output/force_matching/'
                    f'trained_params_alanine_{training_name}.pkl')

used_dataset_size = 500000
train_ratio = 0.8
val_ratio = 0.08
batch_per_device = 500
batch_cache = 50

initial_lr = 0.001
epochs = 100
check_freq = 10

system_temperature = 300  # Kelvin
boltzmann_constant = 0.0083145107  # in kJ / mol K
kbt = system_temperature * boltzmann_constant

model = 'CGDimeNet'

key = random.PRNGKey(0)
model_init_key, shuffle_key = random.split(key, 2)

# build datasets
position_data = data_processing.get_dataset(configuration_str,
                                            retain=used_dataset_size)
force_data = data_processing.get_dataset(force_str, retain=used_dataset_size)

box, _, masses, _ = io.load_box(file_topology)
box_tensor, _ = custom_space.init_fractional_coordinates(box)
displacement, _ = space.periodic_general(box_tensor,
                                         fractional_coordinates=True)

position_data = data_processing.scale_dataset_fractional(position_data,
                                                         box_tensor)
r_init = position_data[0]

lrd = int(used_dataset_size / batch_per_device * epochs)
lr_schedule = optax.exponential_decay(-initial_lr, lrd, 0.01)
optimizer = optax.chain(
    optax.scale_by_adam(),
    optax.scale_by_schedule(lr_schedule)
)

priors = ['bond', 'angle', 'dihedral']
species, prior_idxs, prior_constants = Initialization.select_protein(
    'heavy_alanine_dipeptide', priors)

energy_fn_template, _, init_params, nbrs_init = \
    Initialization.select_model(
        model, r_init, displacement, box, model_init_key, kbt, fractional=True,
        species=species, prior_constants=prior_constants, prior_idxs=prior_idxs)

trainer = trainers.ForceMatching(init_params, energy_fn_template, nbrs_init,
                                 optimizer, position_data,
                                 force_data=force_data,
                                 batch_per_device=batch_per_device,
                                 box_tensor=box_tensor,
                                 batch_cache=batch_cache,
                                 train_ratio=train_ratio,
                                 val_ratio=val_ratio)


trainer.train(epochs)

best_params = trainer.best_params
with open(save_params_path, 'wb') as pickle_file:
    pickle.dump(best_params, pickle_file)

plt.figure()
plt.plot(trainer.train_losses, label='Train', color='#3C5488FF')
plt.plot(trainer.val_losses, label='Val', color='#00A087FF')
plt.legend()
plt.ylabel('MSE Loss')
plt.xlabel('Updates')
plt.savefig(save_plot)
