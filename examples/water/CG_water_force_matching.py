"""Train a CG water model via force matching."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')  # disable warnings about float64 usage

import cloudpickle as pickle
from pathlib import Path

from jax import random, numpy as jnp
from jax_md import space
import matplotlib.pyplot as plt
import numpy as onp
import optax

from chemtrain import trainers, data_processing
from chemtrain.jax_md_mod import custom_space
from util import Initialization

Path('output/figures').mkdir(parents=True, exist_ok=True)
Path('output/force_matching').mkdir(parents=True, exist_ok=True)

# user input
configuration_str = 'data/dataset/conf_COM_10k.npy'
force_str = 'data/dataset/forces_COM_10k.npy'
box_str = 'data/dataset/box.npy'

training_name = 'FM_model_water'
save_plot = f'output/figures/force_matching_losses_{training_name}.png'
save_params_path = ('output/force_matching/'
                    f'trained_params_water_{training_name}.pkl')

used_dataset_size = 10000
train_ratio = 0.8
val_ratio = 0.08
batch_per_device = 10
batch_cache = 10

epochs = 100

box_length = onp.load(box_str)
box = jnp.ones(3) * box_length

model = 'CGDimeNet'
# model = 'Tabulated'

# build datasets
position_data = data_processing.get_dataset(configuration_str,
                                            retain=used_dataset_size)
force_data = data_processing.get_dataset(force_str, retain=used_dataset_size)

dataset_size = position_data.shape[0]
print('Dataset size:', dataset_size)

if model == 'Tabulated':
    initial_lr = 0.1
elif model == 'CGDimeNet':
    initial_lr = 0.001
else:
    raise NotImplementedError

decay_length = int(dataset_size / batch_per_device * epochs)
lr_schedule = optax.exponential_decay(-initial_lr, decay_length, 0.01)
optimizer = optax.chain(
    optax.scale_by_adam(),
    optax.scale_by_schedule(lr_schedule)
)

box_tensor, _ = custom_space.init_fractional_coordinates(box)
displacement, _ = space.periodic_general(box_tensor,
                                         fractional_coordinates=True)
position_data = data_processing.scale_dataset_fractional(position_data,
                                                         box_tensor)
R_init = position_data[0]

model_init_key = random.PRNGKey(0)
constants = {'repulsive': (0.3165, 1., 0.5, 12)}
idxs = {}
energy_fn_template, _, init_params, nbrs_init = \
    Initialization.select_model(model, R_init, displacement, box,
                                model_init_key, fractional=True,
                                prior_constants=constants, prior_idxs=idxs)

trainer = trainers.ForceMatching(init_params, energy_fn_template, nbrs_init,
                                 optimizer, position_data,
                                 force_data=force_data,
                                 batch_per_device=batch_per_device,
                                 box_tensor=box_tensor,
                                 batch_cache=batch_cache,
                                 train_ratio=train_ratio,
                                 val_ratio=val_ratio)

trainer.train(epochs)

with open(save_params_path, 'wb') as pickle_file:
    pickle.dump(trainer.best_params, pickle_file)

plt.figure()
plt.plot(trainer.train_losses, label='Train', color='#3C5488FF')
plt.plot(trainer.val_losses, label='Val', color='#00A087FF')
plt.legend()
plt.ylabel('MSE Loss')
plt.xlabel('Update step')
plt.savefig(save_plot)
