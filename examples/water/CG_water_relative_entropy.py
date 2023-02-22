"""Training a CG water model via relative entropy minimization."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')  # disable warnings about float64 usage

from pathlib import Path

from jax import numpy as jnp
import numpy as onp
import optax

from chemtrain import trainers, traj_util, data_processing
from util import Initialization

Path('output/figures').mkdir(parents=True, exist_ok=True)
Path('output/rel_entropy').mkdir(parents=True, exist_ok=True)

# dataset
configuration_str = 'data/dataset/conf_COM_10k.npy'
box_str = 'data/dataset/box.npy'

save_param_path = 'output/rel_entropy/trained_params_water_RE_model_water.pkl'
used_dataset_size = 8000
num_updates = 300

# simulation parameters
system_temperature = 298  # Kelvin
boltzmann_constant = 0.0083145107  # in kJ / mol K
kbt = system_temperature * boltzmann_constant
mass = 18.0154
rand_seed = 0

time_step = 0.002
total_time = 75.
t_equilib = 5.
print_every = 0.1

model = 'CGDimeNet'
# model = 'Tabulated'

if model == 'Tabulated':
    initial_lr = 0.1
elif model == 'CGDimeNet':
    initial_lr = 0.003
else:
    raise NotImplementedError

lr_schedule = optax.exponential_decay(-initial_lr, num_updates, 0.01)
optimizer = optax.chain(
    optax.scale_by_adam(0.1, 0.4),
    optax.scale_by_schedule(lr_schedule)
)

timings = traj_util.process_printouts(time_step, total_time, t_equilib,
                                      print_every)

box_length = onp.load(box_str)
box = jnp.ones(3) * box_length

position_data = data_processing.get_dataset(configuration_str,
                                            retain=used_dataset_size)
r_init = position_data[0]

constants = {'repulsive': (0.3165, 1., 0.5, 12)}
idxs = {}

simulation_data = Initialization.InitializationClass(
    r_init=r_init, box=box, kbt=kbt, masses=mass, dt=time_step)
reference_state, init_params, simulation_fns, _, _ = \
    Initialization.initialize_simulation(
        simulation_data, model, key_init=rand_seed, prior_constants=constants,
        prior_idxs=idxs)
simulator_template, energy_fn_template, neighbor_fn = simulation_fns

reference_data = data_processing.scale_dataset_fractional(position_data, box)
trainer = trainers.RelativeEntropy(
    init_params, optimizer, energy_fn_template=energy_fn_template,
    reweight_ratio=1.1)

trainer.add_statepoint(reference_data, energy_fn_template, simulator_template,
                       neighbor_fn, timings, kbt, reference_state)

trainer.train(num_updates)

# save parameters
trainer.save_energy_params(save_param_path, '.pkl')
