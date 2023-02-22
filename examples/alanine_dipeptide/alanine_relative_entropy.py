"""Training a CG model of alanine dipeptide via relative entropy minimization.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')  # disable warnings about float64 usage

from pathlib import Path

from jax import random
import optax

from chemtrain import trainers, traj_util, data_processing
from chemtrain.jax_md_mod import io
from util import Initialization

Path('output/figures').mkdir(parents=True, exist_ok=True)
Path('output/rel_entropy').mkdir(parents=True, exist_ok=True)

# user input
configuration_str = 'data/dataset/confs_heavy_100ns.npy'
file_topology = 'data/confs/heavy_2_7nm.gro'

save_params_path = ('output/rel_entropy/'
                    'trained_params_alanine_RE_model_alanine.pkl')

used_dataset_size = 400000
n_trajectory = 50
num_updates = 300

# simulation parameters
system_temperature = 300  # Kelvin
boltzmann_constant = 0.0083145107  # in kJ / mol K
kbt = system_temperature * boltzmann_constant

time_step = 0.002
total_time = 1005
t_equilib = 5.
print_every = 0.2

model = 'CGDimeNet'

initial_lr = 0.003
lr_schedule = optax.exponential_decay(-initial_lr, num_updates, 0.01)
optimizer = optax.chain(
    optax.scale_by_adam(0.1, 0.4),
    optax.scale_by_schedule(lr_schedule)
)

timings = traj_util.process_printouts(time_step, total_time, t_equilib,
                                      print_every)

# initial configuration
box, _, masses, _ = io.load_box(file_topology)

priors = ['bond', 'angle', 'dihedral']
species, prior_idxs, prior_constants = Initialization.select_protein(
    'heavy_alanine_dipeptide', priors)

position_data = data_processing.get_dataset(configuration_str,
                                            retain=used_dataset_size)

# Random starting configurations
key = random.PRNGKey(0)
r_init = random.choice(key, position_data, (n_trajectory,), replace=False)

simulation_data = Initialization.InitializationClass(
    r_init=r_init, box=box, kbt=kbt, masses=masses, dt=time_step,
    species=species)

init_sim_states, init_params, simulation_fns, _, _ = \
    Initialization.initialize_simulation(simulation_data,
                                         model,
                                         integrator='Langevin',
                                         prior_constants=prior_constants,
                                         prior_idxs=prior_idxs)

simulator_template, energy_fn_template, neighbor_fn = simulation_fns

reference_data = data_processing.scale_dataset_fractional(position_data, box)

# a reweight_ratio > 1 disables reweighting
trainer = trainers.RelativeEntropy(init_params, optimizer, reweight_ratio=1.1,
                                   energy_fn_template=energy_fn_template)

trainer.add_statepoint(
    reference_data, energy_fn_template, simulator_template, neighbor_fn,
    timings, kbt, init_sim_states, reference_batch_size=used_dataset_size,
    vmap_batch=n_trajectory)

trainer.train(num_updates)
trainer.save_energy_params(save_params_path, '.pkl')
