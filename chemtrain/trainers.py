"""This file contains several Trainer classes as a quickstart for users."""
from jax import numpy as jnp
from jax_sgmc import data
from jax_sgmc.data import numpy_loader

from chemtrain import (util, force_matching, reweighting,
                       max_likelihood)


class ForceMatching(max_likelihood.DataParallelTrainer):
    """Force-matching trainer.

    This implementation assumes a constant number of particles per box and
    constant box sizes for each snapshot.
    If this is not the case, please use the ForceMatchingPrecomputed trainer
    based on padded sparse neighborlists.
    Caution: Currently neighborlist overflow is not checked.
    Make sure to build nbrs_init large enough.

    Virial data is pressure tensor, i.e. negative stress tensor
    """
    def __init__(self, init_params, energy_fn_template, nbrs_init,
                 optimizer, position_data, energy_data=None, force_data=None,
                 virial_data=None, box_tensor=None, gamma_f=1., gamma_p=1.e-6,
                 batch_per_device=1, batch_cache=10, train_ratio=0.7,
                 val_ratio=0.1, shuffle=False,
                 convergence_criterion='window_median',
                 checkpoint_folder='Checkpoints'):

        checkpoint_path = 'output/force_matching/' + str(checkpoint_folder)
        dataset_dict = {'position_data': position_data,
                        'energy_data': energy_data,
                        'force_data': force_data,
                        'virial_data': virial_data
                        }

        virial_fn = force_matching.init_virial_fn(
            virial_data, energy_fn_template, box_tensor)
        model = force_matching.init_model(nbrs_init, energy_fn_template,
                                          virial_fn)
        loss_fn = force_matching.init_loss_fn(gamma_f=gamma_f, gamma_p=gamma_p)

        super().__init__(dataset_dict, loss_fn, model, init_params, optimizer,
                         checkpoint_path, batch_per_device, batch_cache,
                         train_ratio, val_ratio, shuffle=shuffle,
                         convergence_criterion=convergence_criterion,
                         energy_fn_template=energy_fn_template)
        self._virial_fn = virial_fn
        self._nbrs_init = nbrs_init
        self._init_test_fn()

    @staticmethod
    def _build_dataset(position_data, energy_data=None, force_data=None,
                       virial_data=None):
        return force_matching.build_dataset(position_data, energy_data,
                                            force_data, virial_data)

    def evaluate_mae_testset(self):
        assert self.test_loader is not None, ('No test set available. Check'
                                              ' train and val ratios or add a'
                                              ' test_loader manually.')
        maes = self.mae_fn(self.best_inference_params_replicated)
        for key, mae_value in maes.items():
            print(f'{key}: MAE = {mae_value:.4f}')

    def _init_test_fn(self):
        if self.test_loader is not None:
            self.mae_fn, data_release_fn = force_matching.init_mae_fn(
                self.test_loader, self._nbrs_init,
                self.reference_energy_fn_template, self.batch_size,
                self.batch_cache, self._virial_fn
            )
            self.release_fns.append(data_release_fn)
        else:
            self.mae_fn = None


class RelativeEntropy(reweighting.PropagationBase):
    """Trainer for relative entropy minimization."""
    def __init__(self, init_params, optimizer,
                 reweight_ratio=0.9, sim_batch_size=1, energy_fn_template=None,
                 convergence_criterion='window_median',
                 checkpoint_folder='Checkpoints'):
        """
        Initializes a relative entropy trainer instance.

        Uses first order method optimizer as Hessian is very expensive
        for neural networks. Both reweighting and the gradient formula
        currently assume a NVT ensemble.

        Args:
            init_params: Initial energy parameters
            optimizer: Optimizer from optax
            reweight_ratio: Ratio of reference samples required for n_eff to
                            surpass to allow re-use of previous reference
                            trajectory state. If trajectories should not be
                            re-used, a value > 1 can be specified.
            sim_batch_size: Number of state-points to be processed as a single
                            batch. Gradients will be averaged over the batch
                            before stepping the optimizer.
            energy_fn_template: Function that takes energy parameters and
                                initializes an new energy function. Here, the
                                energy_fn_template is only a reference that
                                will be saved alongside the trainer. Each
                                state point requires its own due to the
                                dependence on the box size via the displacement
                                function, which can vary between state points.
            convergence_criterion: Either 'max_loss' or 'ave_loss'.
                                   If 'max_loss', stops if the gradient norm
                                   across all batches in the epoch is smaller
                                   than convergence_thresh. 'ave_loss' evaluates
                                   the average gradient norm across the batch.
                                   For a single state point, both are
                                   equivalent. A criterion based on the rolling
                                   standard deviation 'std' might be implemented
                                   in the future.
            checkpoint_folder: Name of folders to store ckeckpoints in.
        """

        checkpoint_path = 'output/rel_entropy/' + str(checkpoint_folder)
        init_trainer_state = util.TrainerState(
            params=init_params, opt_state=optimizer.init(init_params))
        super().__init__(init_trainer_state, optimizer, checkpoint_path,
                         reweight_ratio, sim_batch_size, energy_fn_template)

        # in addition to the standard trajectory state, we also need to keep
        # track of dataloader states for reference snapshots
        self.data_states = {}

        self.early_stop = max_likelihood.EarlyStopping(self.params,
                                                       convergence_criterion)

    def _set_dataset(self, key, reference_data, reference_batch_size,
                     batch_cache=1):
        """Set dataset and loader corresponding to current state point."""
        reference_loader = numpy_loader.NumpyDataLoader(R=reference_data,
                                                        copy=False)
        init_reference_batch, get_ref_batch, _ = data.random_reference_data(
            reference_loader, batch_cache, reference_batch_size)
        init_reference_batch_state = init_reference_batch(shuffle=True)
        self.data_states[key] = init_reference_batch_state
        return get_ref_batch

    def add_statepoint(self, reference_data, energy_fn_template,
                       simulator_template, neighbor_fn, timings, kbt,
                       reference_state, reference_batch_size=None,
                       batch_cache=1, initialize_traj=True, set_key=None,
                       vmap_batch=10):
        """
        Adds a state point to the pool of simulations.

        As each reference dataset / trajectory corresponds to a single
        state point, we initialize the dataloader together with the
        simulation.

        Currently only supports NVT simulations.

        Args:
            reference_data: De-correlated reference trajectory
            energy_fn_template: Function that takes energy parameters and
                                initializes an new energy function.
            simulator_template: Function that takes an energy function and
                                returns a simulator function.
            neighbor_fn: Neighbor function
            timings: Instance of TimingClass containing information
                     about the trajectory length and which states to retain
            kbt: Temperature in kbT
            reference_state: Tuple of initial simulation state and neighbor list
            reference_batch_size: Batch size of dataloader for reference
                                  trajectory. If None, will use the same number
                                  of snapshots as generated via the optimizer.
            batch_cache: Number of reference batches to cache in order to
                         minimize host-device communication. Make sure the
                         cached data size does not exceed the full dataset size.
            initialize_traj: True, if an initial trajectory should be generated.
                             Should only be set to False if a checkpoint is
                             loaded before starting any training.
            set_key: Specify a key in order to restart from same statepoint.
                     By default, uses the index of the sequance statepoints are
                     added, i.e. self.trajectory_states[0] for the first added
                     statepoint. Can be used for changing the timings of the
                     simulation during training.
            vmap_batch: Batch size of vmapping of per-snapshot energy and
                        gradient calculation.
        """
        if reference_batch_size is None:
            print('No reference batch size provided. Using number of generated'
                  ' CG snapshots by default.')
            states_per_traj = jnp.size(timings.t_production_start)
            if reference_state[0].position.ndim > 2:
                n_trajctories = reference_state[0].position.shape[0]
                reference_batch_size = n_trajctories * states_per_traj
            else:
                reference_batch_size = states_per_traj

        key, weights_fn, propagate = self._init_statepoint(reference_state,
                                                           energy_fn_template,
                                                           simulator_template,
                                                           neighbor_fn,
                                                           timings,
                                                           kbt,
                                                           set_key,
                                                           vmap_batch,
                                                           initialize_traj)

        reference_dataloader = self._set_dataset(key,
                                                 reference_data,
                                                 reference_batch_size,
                                                 batch_cache)

        grad_fn = reweighting.init_rel_entropy_gradient(
            energy_fn_template, weights_fn, kbt, vmap_batch)

        def propagation_and_grad(params, traj_state, batch_state):
            """Propagates the trajectory, if necessary, and computes the
            gradient via the relative entropy formalism.
            """
            traj_state = propagate(params, traj_state)
            new_batch_state, reference_batch = reference_dataloader(batch_state)
            reference_positions = reference_batch['R']
            grad = grad_fn(params, traj_state, reference_positions)
            return traj_state, grad, new_batch_state

        self.grad_fns[key] = propagation_and_grad

    def _update(self, batch):
        """Updates the potential using the gradient from relative entropy."""
        grads = []
        for sim_key in batch:
            grad_fn = self.grad_fns[sim_key]

            self.trajectory_states[sim_key], curr_grad, \
            self.data_states[sim_key] = grad_fn(self.params,
                                                self.trajectory_states[sim_key],
                                                self.data_states[sim_key])
            grads.append(curr_grad)

        batch_grad = util.tree_mean(grads)
        self._step_optimizer(batch_grad)
        self.gradient_norm_history.append(util.tree_norm(batch_grad))

    def _evaluate_convergence(self, duration, thresh):
        curr_grad_norm = self.gradient_norm_history[-1]
        print(f'\nEpoch {self._epoch}: Elapsed time = {duration:.3f} min')
        self._print_measured_statepoint()

        self._converged = self.early_stop.early_stopping(curr_grad_norm, thresh,
                                                         save_best_params=False)
