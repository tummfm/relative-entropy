"""A collection of functions to initialize Jax, M.D. simulations."""
from functools import partial

import chex
import haiku as hk
from jax import random, vmap, numpy as jnp
from jax_md import util, simulate, partition, space, energy
import numpy as onp
from scipy import interpolate as sci_interpolate

from chemtrain import traj_quantity, layers, neural_networks, dropout
from chemtrain.jax_md_mod import custom_energy, custom_space, custom_quantity

Array = util.Array


@chex.dataclass
class InitializationClass:
    """A dataclass containing initialization information.

    Notes:
      careful: dataclasses.astuple(InitializationClass) sometimes
      changes type from jnp.Array to onp.ndarray

    Attributes:
        r_init: Initial Positions
        box: Simulation box size
        kbT: Target thermostat temperature times Boltzmann constant
        mass: Particle masses
        dt: Time step size
        species: Species index for each particle
        ref_press: Target pressure for barostat
        temperature: Thermostat temperature; only used for computation of
                     thermal expansion coefficient and heat capacity
    """
    r_init: Array
    box: Array
    kbt: float
    masses: Array
    dt: float
    species: Array = None
    ref_press: float = 1.
    temperature: float = None


def select_target_rdf(target_rdf, rdf_start=0., nbins=300):
    if target_rdf == 'TIP4P/2005':
        reference_rdf = onp.loadtxt('data/water_models/'
                                    'TIP4P-2005_300_COM_RDF.csv')
        rdf_cut = 0.85
    else:
        raise ValueError(f'The reference rdf {target_rdf} is not implemented.')

    rdf_bin_centers, rdf_bin_boundaries, sigma_rdf = \
        custom_quantity.rdf_discretization(rdf_cut, nbins, rdf_start)
    rdf_spline = sci_interpolate.interp1d(reference_rdf[:, 0],
                                          reference_rdf[:, 1], kind='cubic')
    reference_rdf = util.f32(rdf_spline(rdf_bin_centers))
    rdf_struct = custom_quantity.RDFParams(reference_rdf, rdf_bin_centers,
                                           rdf_bin_boundaries, sigma_rdf)
    return rdf_struct


def select_target_adf(target_adf, r_outer, r_inner=0., nbins_theta=150):
    if target_adf == 'TIP4P/2005':
        reference_adf = onp.loadtxt('data/water_models/'
                                    'TIP4P-2005_150_COM_ADF.csv')
    else:
        raise ValueError(f'The reference adf {target_adf} is not implemented.')

    adf_bin_centers, sigma_adf = custom_quantity.adf_discretization(nbins_theta)

    adf_spline = sci_interpolate.interp1d(reference_adf[:, 0],
                                          reference_adf[:, 1], kind='cubic')
    reference_adf = util.f32(adf_spline(adf_bin_centers))

    adf_struct = custom_quantity.ADFParams(reference_adf, adf_bin_centers,
                                           sigma_adf, r_outer, r_inner)
    return adf_struct


def select_target_tcf(target_tcf, tcf_cut, tcf_start=0.2, nbins=30):
    if target_tcf == 'TIP4P/2005':
        if tcf_cut == 0.5:
            reference_tcf = onp.load('data/water_models/TIP4P-2005_1k_50b_TCF'
                                     '_cut05.npy')
            dx_bin = 0.3 / nbins
            bins_centers = onp.linspace(0.2 + dx_bin/2., 0.5 - dx_bin/2., 50)
        elif tcf_cut == 0.6:
            reference_tcf = onp.load('data/water_models/TIP4P-2005_1k_50b_TCF'
                                     '_cut06.npy')
            dx_bin = 0.4 / nbins
            bins_centers = onp.linspace(0.2 + dx_bin/2., 0.6 - dx_bin/2., 50)
        elif tcf_cut == 0.8:
            reference_tcf = onp.load('data/water_models/TIP4P-2005_1k_50b_TCF'
                                     '_cut08.npy')
            dx_bin = 0.6 / nbins
            bins_centers = onp.linspace(0.2 + dx_bin/2., 0.8 - dx_bin/2., 50)
        else:
            raise ValueError(f'The cutoff {tcf_cut} is not implemented.')
    else:
        raise ValueError(f'The reference tcf {target_tcf} is not implemented.')

    (sigma_tcf, volume, tcf_x_binx_centers, tcf_y_bin_centers,
     tcf_z_bin_centers) = custom_quantity.tcf_discretization(tcf_cut, nbins,
                                                             tcf_start)

    equilateral = onp.diagonal(onp.diagonal(reference_tcf))
    tcf_spline = sci_interpolate.interp1d(bins_centers, equilateral,
                                          kind='cubic')
    reference_tcf = util.f32(tcf_spline(tcf_x_binx_centers[0, :, 0]))
    tcf_struct = custom_quantity.TCFParams(
        reference_tcf, sigma_tcf, volume, tcf_x_binx_centers, tcf_y_bin_centers,
        tcf_z_bin_centers
    )
    return tcf_struct


def prior_potential(prior_fns, pos, neighbor, **dynamic_kwargs):
    """Evaluates the prior potential for a given snapshot."""
    sum_priors = 0.
    if prior_fns is not None:
        for key in prior_fns:
            sum_priors += prior_fns[key](pos, neighbor=neighbor,
                                         **dynamic_kwargs)
    return sum_priors


def select_priors(displacement, prior_constants, prior_idxs, kbt=None):
    """Build prior potential from combination of classical potentials."""
    prior_fns = {}
    if 'bond' in prior_constants:
        assert kbt is not None, 'Need to provide kbT for bond prior.'
        bond_mean, bond_variance = prior_constants['bond']
        bonds = prior_idxs['bond']
        prior_fns['bond'] = energy.simple_spring_bond(
            displacement, bonds, length=bond_mean, epsilon=kbt / bond_variance)

    if 'angle' in prior_constants:
        assert kbt is not None, 'Need to provide kbT for angle prior.'
        angle_mean, angle_variance = prior_constants['angle']
        angles = prior_idxs['angle']
        prior_fns['angle'] = custom_energy.harmonic_angle(
            displacement, angles, angle_mean, angle_variance, kbt)

    if 'LJ' in prior_constants:
        lj_sigma, lj_epsilon = prior_constants['LJ']
        lj_idxs = prior_idxs['LJ']
        prior_fns['LJ'] = custom_energy.lennard_jones_nonbond(
            displacement, lj_idxs, lj_sigma, lj_epsilon)

    if 'repulsive' in prior_constants:
        re_sigma, re_epsilon, re_cut, re_exp = prior_constants['repulsive']
        prior_fns['repulsive'] = custom_energy.generic_repulsion_neighborlist(
            displacement, sigma=re_sigma, epsilon=re_epsilon, exp=re_exp,
            initialize_neighbor_list=False, r_onset=0.9 * re_cut,
            r_cutoff=re_cut)

    if 'dihedral' in prior_constants:
        dih_phase, dih_constant, dih_n = prior_constants['dihedral']
        dihdral_idxs = prior_idxs['dihedral']
        prior_fns['dihedral'] = custom_energy.periodic_dihedral(
            displacement, dihdral_idxs, dih_phase, dih_constant, dih_n)

    if 'repulsive_nonbonded' in prior_constants:
        # only repulsive part of LJ via idxs instead of nbrs list
        ren_sigma, ren_epsilon = prior_constants['repulsive_nonbonded']
        ren_idxs = prior_idxs['repulsive_nonbonded']
        prior_fns['repulsive_1_4'] = custom_energy.generic_repulsion_nonbond(
            displacement, ren_idxs, sigma=ren_sigma, epsilon=ren_epsilon, exp=6)

    return prior_fns


def select_protein(protein, prior_list):
    idxs = {}
    constants = {}
    if protein == 'heavy_alanine_dipeptide':
        print('Distinguishing different C_Hx atoms')
        species = jnp.array([6, 1, 8, 7, 2, 6, 1, 8, 7, 6])
        if 'bond' in prior_list:
            bond_mean = onp.load('data/prior/Alanine_dipeptide_heavy_eq_bond'
                                 '_length.npy')
            bond_variance = onp.load('data/prior/Alanine_dipeptide_heavy_eq'
                                     '_bond_variance.npy')
            bond_idxs = onp.array([[0, 1],
                                   [1, 2],
                                   [1, 3],
                                   [4, 6],
                                   [6, 7],
                                   [4, 5],
                                   [3, 4],
                                   [6, 8],
                                   [8, 9]])
            idxs['bond'] = bond_idxs
            constants['bond'] = (bond_mean, bond_variance)

        if 'angle' in prior_list:
            angle_mean = onp.load('data/prior/Alanine_dipeptide_heavy_eq'
                                  '_angle.npy')
            angle_variance = onp.load('data/prior/Alanine_dipeptide_heavy_eq'
                                      '_angle_variance.npy')
            angle_idxs = onp.array([[0, 1, 2],
                                    [0, 1, 3],
                                    [2, 1, 3],
                                    [1, 3, 4],
                                    [3, 4, 5],
                                    [3, 4, 6],
                                    [5, 4, 6],
                                    [4, 6, 7],
                                    [4, 6, 8],
                                    [7, 6, 8],
                                    [6, 8, 9]])
            idxs['angle'] = angle_idxs
            constants['angle'] = (angle_mean, angle_variance)

        if 'LJ' in prior_list:
            lj_sigma = onp.load('data/prior/Alanine_dipeptide_heavy_sigma.npy')
            lj_epsilon = onp.load('data/prior/Alanine_dipeptide_heavy_'
                                  'epsilon.npy')
            lj_idxs = onp.array([[0, 5],
                                 [0, 6],
                                 [0, 7],
                                 [0, 8],
                                 [0, 9],
                                 [1, 7],
                                 [1, 8],
                                 [1, 9],
                                 [2, 5],
                                 [2, 6],
                                 [2, 7],
                                 [2, 8],
                                 [2, 9],
                                 [3, 9],
                                 [5, 9]])
            idxs['LJ'] = lj_idxs
            constants['LJ'] = (lj_sigma, lj_epsilon)

        if 'dihedral' in prior_list:
            dihedral_phase = onp.load('data/prior/Alanine_dipeptide_heavy_'
                                      'dihedral_phase.npy')
            dihedral_constant = onp.load('data/prior/Alanine_dipeptide_heavy'
                                         '_dihedral_constant.npy')
            dihedral_n = onp.load('data/prior/Alanine_dipeptide_heavy_dihedral'
                                  '_multiplicity.npy')

            dihedral_idxs = onp.array([[1, 3, 4, 6],
                                       [3, 4, 6, 8],
                                       [0, 1, 3, 4],
                                       [2, 1, 3, 4],
                                       [1, 3, 4, 5],
                                       [5, 4, 6, 8],
                                       [4, 6, 8, 9],
                                       [7, 6, 8, 9]])
            idxs['dihedral'] = dihedral_idxs
            constants['dihedral'] = (dihedral_phase, dihedral_constant,
                                     dihedral_n)

        if 'repulsive_nonbonded' in prior_list:
            # repulsive part of the LJ
            if 'LJ' in prior_list:
                raise ValueError('Not sensible to have LJ and repulsive part of'
                                 ' LJ together. Choose one.')
            ren_sigma = onp.load('data/prior/Alanine_dipeptide_heavy_sigma.npy')
            ren_epsilon = onp.load('data/prior/Alanine_dipeptide'
                                   '_heavy_epsilon.npy')
            ren_idxs = onp.array([[0, 5],
                                  [0, 6],
                                  [0, 7],
                                  [0, 8],
                                  [0, 9],
                                  [1, 7],
                                  [1, 8],
                                  [1, 9],
                                  [2, 5],
                                  [2, 6],
                                  [2, 7],
                                  [2, 8],
                                  [2, 9],
                                  [3, 9],
                                  [5, 9]])
            idxs['repulsive_nonbonded'] = ren_idxs
            constants['repulsive_nonbonded'] = (ren_sigma, ren_epsilon)
    else:
        raise ValueError(f'The protein {protein} is not implemented.')
    return species, idxs, constants


def build_quantity_dict(pos_init, box_tensor, displacement, energy_fn_template,
                        nbrs, target_dict):
    targets = {}
    compute_fns = {}

    if 'rdf' in target_dict:
        rdf_struct = target_dict['rdf']
        rdf_fn = custom_quantity.init_rdf(displacement, rdf_struct, box_tensor)
        rdf_dict = {'target': rdf_struct.reference, 'gamma': 1.,
                    'traj_fn': traj_quantity.init_traj_mean_fn('rdf')}
        targets['rdf'] = rdf_dict
        compute_fns['rdf'] = rdf_fn

    if 'adf' in target_dict:
        adf_struct = target_dict['adf']
        adf_fn = custom_quantity.init_adf_nbrs(
            displacement, adf_struct, smoothing_dr=0.01, r_init=pos_init,
            nbrs_init=nbrs)
        adf_target_dict = {'target': adf_struct.reference, 'gamma': 1.,
                           'traj_fn': traj_quantity.init_traj_mean_fn('adf')}
        targets['adf'] = adf_target_dict
        compute_fns['adf'] = adf_fn

    if 'tcf' in target_dict:
        tcf_struct = target_dict['tcf']
        tcf_fn = custom_quantity.init_tcf_nbrs(displacement, tcf_struct,
                                               box_tensor, nbrs_init=nbrs,
                                               batch_size=1000)
        tcf_target_dict = {'target': tcf_struct.reference, 'gamma': 1.,
                           'traj_fn': traj_quantity.init_traj_mean_fn('tcf')}
        targets['tcf'] = tcf_target_dict
        compute_fns['tcf'] = tcf_fn

    if 'pressure' in target_dict:
        pressure_fn = custom_quantity.init_pressure(energy_fn_template,
                                                    box_tensor)
        pressure_target_dict = {
            'target': target_dict['pressure'], 'gamma': 1.e-7,
            'traj_fn': traj_quantity.init_traj_mean_fn('pressure')}
        targets['pressure'] = pressure_target_dict
        compute_fns['pressure'] = pressure_fn

    if 'pressure_tensor' in target_dict:
        pressure_fn = custom_quantity.init_virial_stress_tensor(
            energy_fn_template, box_tensor)
        pressure_target_dict = {
            'target': target_dict['pressure_tensor'], 'gamma': 1.e-7,
            'traj_fn': traj_quantity.init_traj_mean_fn('pressure_tensor')}
        targets['pressure_tensor'] = pressure_target_dict
        compute_fns['pressure_tensor'] = pressure_fn

    return compute_fns, targets


def default_x_vals(r_cut, delta_cut):
    return jnp.linspace(0.05, r_cut + delta_cut, 100, dtype=jnp.float32)


def select_model(model, init_pos, displacement, box, model_init_key, kbt=None,
                 species=None, x_vals=None, fractional=True,
                 kbt_dependent=False, prior_constants=None, prior_idxs=None,
                 dropout_init_seed=None, **energy_kwargs):
    if model == 'Tabulated':
        r_cut = 0.9
        delta_cut = 0.1
        if x_vals is None:
            x_vals = default_x_vals(r_cut, delta_cut)

        init_params = 0.1 * random.normal(model_init_key, x_vals.shape)
        init_params = jnp.array(init_params, dtype=jnp.float32)
        prior_fn = custom_energy.generic_repulsion_neighborlist(
            displacement, sigma=0.3165, epsilon=1., exp=12,
            initialize_neighbor_list=False, r_onset=0.9 * r_cut,
            r_cutoff=r_cut)

        tabulated_energy = partial(
            custom_energy.tabulated_neighbor_list, displacement, x_vals,
            box_size=box, r_onset=(r_cut - 0.2), r_cutoff=r_cut,
            dr_threshold=0.05, capacity_multiplier=1.25
        )
        neighbor_fn, _ = tabulated_energy(init_params)

        nbrs_init = neighbor_fn.allocate(init_pos, extra_capacity=0)

        def energy_fn_template(energy_params):
            tab_energy = tabulated_energy(energy_params,
                                          initialize_neighbor_list=False)

            def energy_fn(pos, neighbor, **dynamic_kwargs):
                return (tab_energy(pos, neighbor, **dynamic_kwargs)
                        + prior_fn(pos, neighbor=neighbor, **dynamic_kwargs)
                        )
            return energy_fn

    elif model == 'CGDimeNet':
        r_cut = 0.5
        n_species = 10

        mlp_init = {
            'b_init': hk.initializers.Constant(0.),
            'w_init': layers.OrthogonalVarianceScalingInit(scale=1.)
        }

        neighbor_fn = partition.neighbor_list(displacement, box, r_cut,
                                              dr_threshold=0.05,
                                              capacity_multiplier=1.5,
                                              fractional_coordinates=fractional,
                                              disable_cell_list=True)

        # create neighborlist for init of GNN
        nbrs_init = neighbor_fn.allocate(init_pos, extra_capacity=0)

        if prior_constants is not None:
            prior_fns = select_priors(displacement, prior_constants, prior_idxs,
                                      kbt)
            print('Using the following priors:')
            [print(key) for key in prior_fns]
        else:
            print('Using no priors')
            prior_fns = None

        dropout_mode = {'output': 0.1, 'interaction': 0.1, 'embedding': 0.1}

        init_fn, gnn_energy_fn = neural_networks.dimenetpp_neighborlist(
            displacement, r_cut, n_species, init_pos, nbrs_init,
            kbt_dependent=kbt_dependent, embed_size=32, init_kwargs=mlp_init,
            dropout_mode=dropout_mode
        )

        # needs to know positions to know shape for network init
        if isinstance(model_init_key, list):
            # ensemble of neural networks not needed together with dropout
            init_params = [init_fn(key, init_pos, neighbor=nbrs_init,
                                   species=species, **energy_kwargs)
                           for key in model_init_key]
        else:
            if dropout_init_seed is None:
                init_params = init_fn(model_init_key, init_pos,
                                      neighbor=nbrs_init,
                                      species=species, **energy_kwargs)
            else:
                dropout_init_key = random.PRNGKey(dropout_init_seed)
                init_params = init_fn(model_init_key, init_pos,
                                      neighbor=nbrs_init, species=species,
                                      dropout_key=dropout_init_key,
                                      **energy_kwargs)
                init_params = dropout.build_dropout_params(init_params,
                                                           dropout_init_key)

        # this pattern allows changing the energy parameters on-the-fly
        def energy_fn_template(energy_params):
            def energy_fn(pos, neighbor, **dynamic_kwargs):
                gnn_energy = gnn_energy_fn(energy_params, pos, neighbor,
                                           species=species, **dynamic_kwargs)
                prior_energy = prior_potential(prior_fns, pos, neighbor,
                                               **dynamic_kwargs)
                return gnn_energy + prior_energy
            return energy_fn

    else:
        raise ValueError('The model' + model + 'is not implemented.')

    return energy_fn_template, neighbor_fn, init_params, nbrs_init


def initialize_simulation(init_class, model, target_dict=None, x_vals=None,
                          key_init=0, fractional=True, integrator='Nose_Hoover',
                          wrapped=True, kbt_dependent=False,
                          prior_constants=None, prior_idxs=None,
                          dropout_init_seed=None):
    key = random.PRNGKey(key_init)
    model_init_key, simulation_init_key = random.split(key, 2)

    box = init_class.box
    box_tensor, scale_fn = custom_space.init_fractional_coordinates(box)
    r_inits = init_class.r_init

    if fractional:
        r_inits = scale_fn(r_inits)

    multi_trajectory = r_inits.ndim > 2
    init_pos = r_inits[0] if multi_trajectory else r_inits

    displacement, shift = space.periodic_general(
        box_tensor, fractional_coordinates=fractional, wrapped=wrapped)

    energy_kwargs = {}
    if kbt_dependent:
        # to allow init of kbt_embedding
        energy_kwargs['kT'] = init_class.kbt

    energy_fn_template, neighbor_fn, init_params, nbrs = select_model(
        model, init_pos, displacement, box, model_init_key, init_class.kbt,
        init_class.species, x_vals, fractional, kbt_dependent,
        prior_idxs=prior_idxs, prior_constants=prior_constants,
        dropout_init_seed=dropout_init_seed, **energy_kwargs
    )

    energy_fn_init = energy_fn_template(init_params)

    # setup simulator
    if integrator == 'Nose_Hoover':
        simulator_template = partial(simulate.nvt_nose_hoover, shift_fn=shift,
                                     dt=init_class.dt, kT=init_class.kbt,
                                     chain_length=3, chain_steps=1)
    elif integrator == 'Langevin':
        simulator_template = partial(simulate.nvt_langevin, shift_fn=shift,
                                     dt=init_class.dt, kT=init_class.kbt,
                                     gamma=100.)

    elif integrator == 'NVE':
        simulator_template = partial(simulate.nve, shift_fn=shift,
                                     dt=init_class.dt)
    else:
        raise NotImplementedError('Integrator string not recognized!')

    init, _ = simulator_template(energy_fn_init)

    if integrator == 'NVE':
        init = partial(init, kT=init_class.kbt)

    def init_sim_state(rng_key, pos):
        nbrs_update = nbrs.update(pos)
        state = init(rng_key, pos, mass=init_class.masses, neighbor=nbrs_update,
                     box=box_tensor, **energy_kwargs)
        return state, nbrs_update  # store together

    if multi_trajectory:
        n_inits = r_inits.shape[0]
        init_keys = random.split(simulation_init_key, n_inits)
        sim_state = vmap(init_sim_state)(init_keys, r_inits)
    else:
        sim_state = init_sim_state(simulation_init_key, init_pos)

    if target_dict is None:
        target_dict = {}
    compute_fns, targets = build_quantity_dict(
        init_pos, box_tensor, displacement, energy_fn_template, nbrs,
        target_dict)

    simulation_funs = (simulator_template, energy_fn_template, neighbor_fn)
    return sim_state, init_params, simulation_funs, compute_fns, targets
