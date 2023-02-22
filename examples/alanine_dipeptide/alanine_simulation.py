"""Forward simulation of alanine dipeptide."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')  # disable warnings about float64 usage

import cloudpickle as pickle
from pathlib import Path
import time

from jax import vmap, random, tree_util, numpy as jnp
from jax_md import space
import numpy as onp

from chemtrain.jax_md_mod import io, custom_space, custom_quantity
from chemtrain import data_processing, traj_util
from util import Initialization
import visualization

Path('output/trajectories').mkdir(parents=True, exist_ok=True)

# user input
save_name = 'FM_2fs_100ns'
folder_name = 'FM_2fs_100ns/'
labels = ['Reference', 'Predicted']

file_topology = 'data/confs/heavy_2_7nm.gro'
configuration_str = 'data/dataset/confs_heavy_100ns.npy'
used_dataset_size = 500000
n_trajectory = 50

model = 'CGDimeNet'


saved_params_path = ('output/force_matching/'
                     'trained_params_alanine_FM_model_alanine.pkl')
# saved_params_path = ('output/rel_entropy/'
#                      'trained_params_alanine_RE_model_alanine.pkl')

# simulation parameters
system_temperature = 300  # Kelvin
boltzmann_constant = 0.0083145107  # in kJ / mol K
kbt = system_temperature * boltzmann_constant
time_step = 0.002

total_time = 110000
t_equilib = 10000.
print_every = 0.2
###############################

box, _, masses, _ = io.load_box(file_topology)
priors = ['bond', 'angle', 'dihedral']
species, prior_idxs, prior_constants = Initialization.select_protein(
    'heavy_alanine_dipeptide', priors)

# random starting points
position_data = data_processing.get_dataset(configuration_str)[1:]
key = random.PRNGKey(0)
r_init = random.choice(key, position_data, (n_trajectory,), replace=False)

simulation_data = Initialization.InitializationClass(
    r_init=r_init, box=box, kbt=kbt, masses=masses, dt=time_step,
    species=species)
timings = traj_util.process_printouts(time_step, total_time, t_equilib,
                                      print_every)

reference_state, energy_params, simulation_fns, compute_fns, targets = \
    Initialization.initialize_simulation(simulation_data,
                                         model,
                                         integrator='Langevin',
                                         prior_constants=prior_constants,
                                         prior_idxs=prior_idxs)

simulator_template, energy_fn_template, neighbor_fn = simulation_fns

if saved_params_path is not None:
    print('Using saved params')
    with open(saved_params_path, 'rb') as pickle_file:
        params = pickle.load(pickle_file)
    energy_params = tree_util.tree_map(jnp.array, params)

trajectory_generator = traj_util.trajectory_generator_init(simulator_template,
                                                           energy_fn_template,
                                                           timings)

t_start = time.time()
traj_state = trajectory_generator(energy_params, reference_state)
t_end = time.time() - t_start
print('total runtime in min:', t_end / 60.)

assert not traj_state.overflow, ('Neighborlist overflow during trajectory '
                                 'generation. Increase capacity and re-run.')

# postprocessing
traj_positions = traj_state.trajectory.position
jnp.save(f'output/trajectories/confs_alanine_{save_name}', traj_positions)

box_tensor, _ = custom_space.init_fractional_coordinates(box)
displacement, _ = space.periodic_general(box_tensor,
                                         fractional_coordinates=True)
position_data = data_processing.scale_dataset_fractional(position_data, box)

# dihedrals
dihedral_idxs = jnp.array([[1, 3, 4, 6], [3, 4, 6, 8]])  # 0: phi    1: psi
batched_dihedrals = vmap(custom_quantity.dihedral_displacement, (0, None, None))

dihedrals_ref = batched_dihedrals(position_data, displacement, dihedral_idxs)
dihedral_angles = batched_dihedrals(traj_positions, displacement, dihedral_idxs)

phi = dihedral_angles[:, 0].reshape((n_trajectory, -1))
psi = dihedral_angles[:, 1].reshape((n_trajectory, -1))

# mean squared error (in rad)
nbins = 60
dihedrals_ref_rad = jnp.deg2rad(dihedrals_ref)
dihedral_angles_rad = jnp.deg2rad(dihedral_angles)
h_ref, y1, y2 = onp.histogram2d(
    dihedrals_ref_rad[:, 0], dihedrals_ref_rad[:, 1], bins=nbins, density=True)
h_pred, _, _ = onp.histogram2d(
    dihedral_angles_rad[:, 0], dihedral_angles_rad[:, 1], bins=nbins,
    density=True)

mse = onp.mean((h_ref - h_pred)**2)
print('MSE of the phi-psi dihedral density histrogram: ', mse)

# unstack parallel trajectories
dihedral_angles_split = dihedral_angles.reshape((n_trajectory, -1, 2))

# Plots
phi_angles_ref = onp.load('data/dataset/phi_angles_r100ns.npy')
psi_angles_ref = onp.load('data/dataset/psi_angles_r100ns.npy')

# dihedral histograms
Path(f'output/postprocessing/{folder_name}').mkdir(parents=True, exist_ok=True)


visualization.plot_histogram_density(dihedral_angles_split[0, :],
                                     save_name + '_first_predicted_',
                                     folder=folder_name)
visualization.plot_histogram_density(dihedrals_ref, save_name + '_REF',
                                     folder=folder_name)

visualization.plot_1d_dihedral(
    [phi_angles_ref, phi], 'phi_' + save_name, labels=labels[0:2],
    folder=folder_name)
visualization.plot_1d_dihedral(
    [psi_angles_ref, psi], 'psi_' + save_name, location='upper left',
    labels=labels[0:2], xlabel='$\psi$ in deg', folder=folder_name)

visualization.plot_histogram_free_energy(dihedral_angles_split[0, :],
                                         save_name + '_pred',
                                         kbt, folder=folder_name)
visualization.plot_histogram_free_energy(dihedrals_ref, save_name + '_REF',
                                         kbt, folder=folder_name)

visualization.plot_compare_histogram_free_energy(
    [dihedrals_ref, dihedral_angles_split[0, :]], save_name, kbt, titles=labels,
    folder=folder_name)

visualization.plot_compare_histogram_density(
    [dihedrals_ref, dihedral_angles_split[0, :]], save_name, titles=labels,
    folder=folder_name)
