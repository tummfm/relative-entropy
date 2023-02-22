"""Implementation of the reweighting formalism.

Allows re-using existing trajectories.
p(S) = ...
"""
from abc import abstractmethod
import time
import warnings

from jax import (checkpoint, lax, jit, random, grad, tree_util, vmap,
                 numpy as jnp)
from jax_md import util as jax_md_util

from chemtrain import util, traj_util, max_likelihood
from chemtrain.jax_md_mod import custom_quantity


def checkpoint_quantities(compute_fns):
    """Applies checkpoint to all compute_fns to save memory on backward pass."""
    for quantity_key in compute_fns:
        compute_fns[quantity_key] = checkpoint(compute_fns[quantity_key])


def _estimate_effective_samples(weights):
    """Returns the effective sample size after reweighting to
    judge reweighting quality.
    """
    # mask to avoid NaN from log(0) if a few weights are 0.
    weights = jnp.where(weights > 1.e-10, weights, 1.e-10)
    exponent = -jnp.sum(weights * jnp.log(weights))
    return jnp.exp(exponent)


def _build_weights(exponents):
    """Returns weights and the effective sample size from exponents
    of the reweighting formulas in a numerically stable way.
    """

    # The reweighting scheme is a softmax, where the exponent above
    # represents the logits. To improve numerical stability and
    # guard against overflow it is good practice to subtract the
    # max of the exponent using the identity softmax(x + c) =
    # softmax(x). With all values in the exponent <=0, this
    # rules out overflow and the 0 value guarantees a denominator >=1.
    exponents -= jnp.max(exponents)
    prob_ratios = jnp.exp(exponents)
    weights = prob_ratios / jax_md_util.high_precision_sum(prob_ratios)
    n_eff = _estimate_effective_samples(weights)
    return weights, n_eff


def init_pot_reweight_propagation_fns(energy_fn_template, simulator_template,
                                      neighbor_fn, timings, ref_kbt,
                                      ref_press=None, reweight_ratio=0.9,
                                      npt_ensemble=False, energy_batch_size=10):
    """
    Initializes all functions necessary for trajectory reweighting for
    a single state point.

    Initialized functions include a function that computes weights for a
    given trajectory and a function that propagates the trajectory forward
    if the statistical error does not allow a re-use of the trajectory.
    The propagation function also ensures that generated trajectories
    did not encounter any neighbor list overflow.
    """
    traj_energy_fn = custom_quantity.energy_wrapper(energy_fn_template)
    reweighting_quantities = {'energy': traj_energy_fn}

    if npt_ensemble:
        # pressure currently only used to print pressure of generated trajectory
        # such that user can ensure correct statepoint of reference trajectory
        pressure_fn = custom_quantity.init_pressure(energy_fn_template)
        reweighting_quantities['pressure'] = pressure_fn

    trajectory_generator = traj_util.trajectory_generator_init(
        simulator_template, energy_fn_template, timings, reweighting_quantities)

    beta = 1. / ref_kbt
    checkpoint_quantities(reweighting_quantities)

    def compute_weights(params, traj_state):
        """Computes weights for the reweighting approach."""

        # reweighting properties (U and pressure) under perturbed potential
        reweight_properties = traj_util.quantity_traj(
            traj_state, reweighting_quantities, params, energy_batch_size)

        # Note: Difference in pot. Energy is difference in total energy
        # as kinetic energy is the same and cancels
        exponent = -beta * (reweight_properties['energy']
                            - traj_state.aux['energy'])
        return _build_weights(exponent)

    def trajectory_identity_mapping(inputs):
        """Re-uses trajectory if no recomputation needed."""
        traj_state = inputs[1]
        return traj_state

    def recompute_trajectory(inputs):
        """Recomputes the reference trajectory, starting from the last
        state of the previous trajectory to save equilibration time.
        """
        params, traj_state = inputs
        # give kT here as additional input to be handed through to energy_fn
        # for kbt-dependent potentials
        updated_traj = trajectory_generator(params, traj_state.sim_state,
                                            kT=ref_kbt, pressure=ref_press)
        return updated_traj

    @jit
    def propagation_fn(params, traj_state):
        """Checks if a trajectory can be re-used. If not, a new trajectory
        is generated ensuring trajectories are always valid.
        Takes params and the traj_state as input and returns a
        trajectory valid for reweighting as well as an error code
        indicating if the neighborlist buffer overflowed during trajectory
        generation.
        """
        _, n_eff = compute_weights(params, traj_state)
        n_snapshots = traj_state.aux['energy'].size
        recompute = n_eff < reweight_ratio * n_snapshots
        propagated_state = lax.cond(recompute,
                                    recompute_trajectory,
                                    trajectory_identity_mapping,
                                    (params, traj_state))
        return propagated_state

    def safe_propagate_traj(params, traj_state):
        """Recomputes trajectory and neighbor list until a trajectory without
        overflow was obtained.
        """
        reset_counter = 0
        while traj_state.overflow:
            warnings.warn('Neighborlist buffer overflowed. '
                          'Initializing larger neighborlist.')
            if reset_counter == 3:  # still overflow after multiple resets
                raise RuntimeError('Multiple neighbor list re-computations did '
                                   'not yield a trajectory without overflow. '
                                   'Consider increasing the neighbor list '
                                   'capacity multiplier.')

            # Note: We restart the simulation from the last trajectory, where
            # we know that no overflow has occured. Re-starting from a state
            # that was generated with overflow is dangerous as overflown
            # particles could cause exploding forces once re-considered.
            last_state, _ = traj_state.sim_state
            if jnp.any(jnp.isnan(last_state.position)):
                raise RuntimeError('Last state is NaN. Currently there is no '
                                   'recovering from this. Restart from the last'
                                   ' non-overflown state might help, but comes'
                                   ' at the cost that the reference state is '
                                   'likely not representative.')
            if last_state.position.ndim > 2:
                single_enlarged_nbrs = util.neighbor_allocate(
                    neighbor_fn, util.tree_get_single(last_state))
                enlarged_nbrs = vmap(util.neighbor_update, (None, 0))(
                    single_enlarged_nbrs, last_state)
            else:
                enlarged_nbrs = util.neighbor_allocate(neighbor_fn, last_state)
            reset_traj_state = traj_state.replace(
                sim_state=(last_state, enlarged_nbrs))
            traj_state = recompute_trajectory((params, reset_traj_state))
            reset_counter += 1
        return traj_state

    def propagate(params, old_traj_state):
        """Wrapper around jitted propagation function that ensures that
        if neighbor list buffer overflowed, the trajectory is recomputed and
        the neighbor list size is increased until valid trajectory was obtained.
        Due to the recomputation of the neighbor list, this function cannot be
        jit.
        """
        new_traj_state = propagation_fn(params, old_traj_state)
        new_traj_state = safe_propagate_traj(params, new_traj_state)
        return new_traj_state

    def init_first_traj(params, reference_state):
        """Initializes initial trajectory to start optimization from.

        We dump the initial trajectory for equilibration, as initial
        equilibration usually takes much longer than equilibration time
        of each trajectory. If this is still not sufficient, the simulation
        should equilibrate over the course of subsequent updates.
        """
        dump_traj = trajectory_generator(params, reference_state,
                                         kT=ref_kbt, pressure=ref_press)

        t_start = time.time()
        init_traj = trajectory_generator(params, dump_traj.sim_state,
                                         kT=ref_kbt, pressure=ref_press)
        runtime = (time.time() - t_start) / 60.  # in mins
        init_traj = safe_propagate_traj(params, init_traj)
        return init_traj, runtime

    return init_first_traj, compute_weights, propagate


def init_rel_entropy_gradient(energy_fn_template, compute_weights, kbt,
                              vmap_batch_size=10):
    """Initializes a function that computes the relative entropy gradient.

    The computation of the gradient is batched to increase computational
    efficiency.

    Args:
        energy_fn_template: Energy function template
        compute_weights: compute_weights function as initialized from
                         init_pot_reweight_propagation_fns.
        kbt: KbT
        vmap_batch_size: Batch size for

    Returns:
        A function rel_entropy_gradient(params, traj_state, reference_batch),
        which returns the relative entropy gradient of 'params' given a
        generated trajectory saved in 'traj_state' and a reference trajectory
        'reference_batch'.
    """
    beta = 1 / kbt

    @jit
    def rel_entropy_gradient(params, traj_state, reference_batch):
        if traj_state.sim_state[0].position.ndim > 2:
            nbrs_init = util.tree_get_single(traj_state.sim_state[1])
        else:
            nbrs_init = traj_state.sim_state[1]

        def energy(params, position):
            energy_fn = energy_fn_template(params)
            # Note: nbrs update requires constant box, i.e. not yet
            # applicable to npt ensemble
            nbrs = nbrs_init.update(position)
            return energy_fn(position, neighbor=nbrs)

        def weighted_gradient(map_input):
            position, weight = map_input
            snapshot_grad = grad(energy)(params, position)  # dudtheta
            weight_gradient = lambda new_grad: weight * new_grad
            weighted_grad_snapshot = tree_util.tree_map(weight_gradient,
                                                        snapshot_grad)
            return weighted_grad_snapshot

        def add_gradient(map_input):
            batch_gradient = vmap(weighted_gradient)(map_input)
            return util.tree_sum(batch_gradient, axis=0)

        weights, _ = compute_weights(params, traj_state)

        # reshape for batched computations
        batch_weights = weights.reshape((-1, vmap_batch_size))
        traj_shape = traj_state.trajectory.position.shape
        batchwise_gen_traj = traj_state.trajectory.position.reshape(
            (-1, vmap_batch_size, traj_shape[-2], traj_shape[-1]))
        ref_shape = reference_batch.shape
        reference_batches = reference_batch.reshape(
            (-1, vmap_batch_size, ref_shape[-2], ref_shape[-1]))

        # no reweighting for reference data: weights = 1 / N
        ref_weights = jnp.ones(reference_batches.shape[:2]) / (ref_shape[0])

        ref_grad = lax.map(add_gradient, (reference_batches, ref_weights))
        mean_ref_grad = util.tree_sum(ref_grad, axis=0)
        gen_traj_grad = lax.map(add_gradient, (batchwise_gen_traj,
                                               batch_weights))
        mean_gen_grad = util.tree_sum(gen_traj_grad, axis=0)

        combine_grads = lambda x, y: beta * (x - y)
        dtheta = tree_util.tree_map(combine_grads, mean_ref_grad, mean_gen_grad)
        return dtheta
    return rel_entropy_gradient


class PropagationBase(max_likelihood.MLETrainerTemplate):
    """Trainer base class for shared functionality whenever (multiple)
    simulations are run during training. Can be used as a template to
    build other trainers. Currently used for DiffTRe and relative entropy.

    We only save the latest generated trajectory for each state point.
    While accumulating trajectories would enable more frequent reweighting,
    this effect is likely minor as past trajectories become exponentially
    less useful with changing potential. Additionally, saving long trajectories
    for each statepoint would increase memory requirements over the course of
    the optimization.
    """
    def __init__(self, init_trainer_state, optimizer, checkpoint_path,
                 reweight_ratio=0.9, sim_batch_size=1, energy_fn_template=None):
        super().__init__(optimizer, init_trainer_state, checkpoint_path,
                         energy_fn_template)
        self.sim_batch_size = sim_batch_size
        self.reweight_ratio = reweight_ratio

        # store for each state point corresponding traj_state and grad_fn
        # save in distinct dicts as grad_fns need to be deleted for checkpoint
        self.grad_fns, self.trajectory_states, self.statepoints = {}, {}, {}
        self.n_statepoints = 0
        self.shuffle_key = random.PRNGKey(0)

    def _init_statepoint(self, reference_state, energy_fn_template,
                         simulator_template, neighbor_fn, timings, kbt,
                         set_key=None, energy_batch_size=10,
                         initialize_traj=True, ref_press=None):
        """Initializes the simulation and reweighting functions as well
        as the initial trajectory for a statepoint."""
        if set_key is not None:
            key = set_key
            if set_key not in self.statepoints.keys():
                self.n_statepoints += 1
        else:
            key = self.n_statepoints
            self.n_statepoints += 1
        self.statepoints[key] = {'kbT': kbt}
        npt_ensemble = util.is_npt_ensemble(reference_state[0])
        if npt_ensemble: self.statepoints[key]['pressure'] = ref_press

        gen_init_traj, compute_weights, propagate = \
            init_pot_reweight_propagation_fns(energy_fn_template,
                                              simulator_template,
                                              neighbor_fn,
                                              timings,
                                              kbt,
                                              ref_press,
                                              self.reweight_ratio,
                                              npt_ensemble,
                                              energy_batch_size)
        if initialize_traj:
            init_traj, runtime = gen_init_traj(self.params, reference_state)
            print(f'Time for trajectory initialization {key}: {runtime} mins')
            self.trajectory_states[key] = init_traj
        else:
            print('Not initializing the initial trajectory is only valid if '
                  'a checkpoint is loaded. In this case, please be use to add '
                  'state points in the same sequence, otherwise loaded '
                  'trajectories will not match its respective simulations.')

        return key, compute_weights, propagate

    @abstractmethod
    def add_statepoint(self, *args, **kwargs):
        """User interface to add additional state point to train model on."""
        raise NotImplementedError()

    @property
    def params(self):
        return self.state.params

    @params.setter
    def params(self, loaded_params):
        self.state = self.state.replace(params=loaded_params)

    def get_sim_state(self, key):
        return self.trajectory_states[key].sim_state

    def _get_batch(self):
        """Helper function to re-shuffle simulations and split into batches."""
        self.shuffle_key, used_key = random.split(self.shuffle_key, 2)
        shuffled_indices = random.permutation(used_key, self.n_statepoints)
        if self.sim_batch_size == 1:
            batch_list = jnp.split(shuffled_indices, shuffled_indices.size)
        elif self.sim_batch_size == -1:
            batch_list = jnp.split(shuffled_indices, 1)
        else:
            raise NotImplementedError('Only batch_size = 1 or -1 implemented.')

        return (batch.tolist() for batch in batch_list)

    def _print_measured_statepoint(self):
        """Print meausured kbT (and pressure for npt ensemble) for all
        statepoints to ensure the simulation is indeed carried out at the
        prescribed state point.
        """
        for sim_key, traj in self.trajectory_states.items():
            statepoint = self.statepoints[sim_key]
            measured_kbt = jnp.mean(traj.aux['kbT'])
            if 'pressure' in statepoint:  # NPT
                measured_press = jnp.mean(traj.aux['pressure'])
                press_print = (f' press = {measured_press:.2f} ref_press = '
                               f'{statepoint["pressure"]:.2f}')
            else:
                press_print = ''
            print(f'Statepoint {sim_key}: kbT = {measured_kbt:.3f} ref_kbt = '
                  f'{statepoint["kbT"]:.3f}' + press_print)

    def train(self, max_epochs, thresh=None, checkpoint_freq=None):
        assert self.n_statepoints > 0, ('Add at least 1 state point via '
                                        '"add_statepoint" to start training.')
        super().train(max_epochs, thresh=thresh,
                      checkpoint_freq=checkpoint_freq)

    @abstractmethod
    def _update(self, batch):
        """Implementation of gradient computation, stepping of the optimizer
        and logging of auxiliary results. Takes batch of simulation indices
        as input.
        """
