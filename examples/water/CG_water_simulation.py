"""Runs a CG water simulation in Jax M.D with loaded parameters.
Trajectory generation for postprocessing and analysis of simulations.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')  # disable warnings about float64 usage

from pathlib import Path
import time

import cloudpickle as pickle
from jax import tree_util, numpy as jnp
import numpy as onp

from chemtrain import traj_util, data_processing
from util import Postprocessing, Initialization

Path('output/figures').mkdir(parents=True, exist_ok=True)
Path('output/trajectories').mkdir(parents=True, exist_ok=True)
Path('output/properties').mkdir(parents=True, exist_ok=True)

configuration_str = 'data/dataset/conf_COM_10k.npy'
box_str = 'data/dataset/box.npy'

# model = 'Tabulated'
model = 'CGDimeNet'

plotname = 'FM_2fs_1ns'


saved_params_path = 'output/rel_entropy/trained_params_water_RE_model_water.pkl'
# saved_params_path = ('output/force_matching/'
#                      'trained_params_water_FM_model_water.pkl')

system_temperature = 298  # Kelvin
boltzmann_constant = 0.0083145107  # in kJ / mol K
kbt = system_temperature * boltzmann_constant
mass = 18.0154
time_step = 0.002

total_time = 1100.
t_equilib = 100.
print_every = 0.1

target_name = 'TIP4P/2005'
rdf_struct = Initialization.select_target_rdf(target_name)
tcf_struct = Initialization.select_target_tcf(target_name, 0.5, nbins=50)
adf_struct = Initialization.select_target_adf(target_name, 0.318)

# targets are only dummy
target_dict = {'rdf': rdf_struct, 'adf': adf_struct, 'tcf': tcf_struct,
               'pressure': 1.}

###############################
used_dataset_size = 1000
box_length = onp.load(box_str)
box = jnp.ones(3) * box_length

position_data = data_processing.get_dataset(configuration_str,
                                            retain=used_dataset_size)
r_init = onp.array(position_data[0])

constants = {'repulsive': (0.3165, 1., 0.5, 12)}
idxs = {}

simulation_data = Initialization.InitializationClass(
    r_init=r_init, box=box, kbt=kbt, masses=mass, dt=time_step)
timings = traj_util.process_printouts(time_step, total_time, t_equilib,
                                      print_every)

reference_state, energy_params, simulation_fns, compute_fns, _ = \
    Initialization.initialize_simulation(
        simulation_data, model, target_dict,
        prior_idxs=idxs, prior_constants=constants
    )
simulator_template, energy_fn_template, neighbor_fn = simulation_fns

if saved_params_path is not None:
    with open(saved_params_path, 'rb') as pickle_file:
        params = pickle.load(pickle_file)
        energy_params = tree_util.tree_map(jnp.array, params)

trajectory_generator = traj_util.trajectory_generator_init(simulator_template,
                                                           energy_fn_template,
                                                           timings)

# compute trajectory and quantities
t_start = time.time()
traj_state = trajectory_generator(energy_params, reference_state)
print('trajectory ps/min: ', total_time / ((time.time() - t_start) / 60.))

jnp.save(f'output/trajectories/CG_water_trajectory_{plotname}',
         traj_state.trajectory.position)
assert not traj_state.overflow, ('Neighborlist overflow during trajectory '
                                 'generation. Increase capacity and re-run.')

print('average kbT:', jnp.mean(traj_state.aux['kbT']), 'vs reference:', kbt)

t_post_start = time.time()
quantity_trajectory = traj_util.quantity_traj(traj_state, compute_fns,
                                              energy_params, batch_size=2)
print('quantity runtime in min: ', (time.time() - t_post_start) / 60.)

if 'rdf' in quantity_trajectory:
    computed_RDF = jnp.mean(quantity_trajectory['rdf'], axis=0)
    jnp.save(f'output/properties/{plotname}_RDF',
             jnp.array([rdf_struct.rdf_bin_centers, computed_RDF]).T)
    Postprocessing.plot_initial_and_predicted_rdf(rdf_struct.rdf_bin_centers,
                                                  computed_RDF, model,
                                                  plotname,
                                                  rdf_struct.reference)

if ('pressure_tensor' in quantity_trajectory
        or 'pressure' in quantity_trajectory):
    if 'pressure_tensor' in quantity_trajectory:
        pressure_traj = quantity_trajectory['pressure_tensor']
    else:
        pressure_traj = quantity_trajectory['pressure']
    mean_pressure = jnp.mean(pressure_traj, axis=0)
    std_pressure = jnp.std(pressure_traj, axis=0)
    print('Pressure scalar mean:', mean_pressure, 'and standard deviation:',
          std_pressure)

if 'adf' in quantity_trajectory:
    computed_ADF = jnp.mean(quantity_trajectory['adf'], axis=0)
    jnp.save(f'output/properties/{plotname}_ADF',
             jnp.array([adf_struct.adf_bin_centers, computed_ADF]).T)
    Postprocessing.plot_initial_and_predicted_adf(adf_struct.adf_bin_centers,
                                                  computed_ADF, model,
                                                  plotname,
                                                  adf_struct.reference)

if 'tcf' in quantity_trajectory:
    computed_TCF = jnp.mean(quantity_trajectory['tcf'], axis=0)
    equilateral = jnp.diagonal(jnp.diagonal(computed_TCF))
    jnp.save(f'output/properties/{plotname}_TCF',
             jnp.array([tcf_struct.tcf_x_bin_centers[0, :, 0], equilateral]).T)
    Postprocessing.plot_initial_and_predicted_tcf(
        tcf_struct.tcf_x_bin_centers[0, :, 0], equilateral, model, plotname,
        tcf_struct.reference)
