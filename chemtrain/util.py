"""Utility functions helpful in designing new trainers."""
import abc
from functools import partial
import pathlib
from typing import Any

import chex
import cloudpickle as pickle
from jax import tree_map, device_count, numpy as jnp
from jax.tree_util import tree_flatten, tree_leaves
from jax_md import simulate
import numpy as onp


# freezing seems to give slight performance improvement
@partial(chex.dataclass, frozen=True)
class TrainerState:
    """Each trainer at least contains the state of parameter and
    optimizer.
    """
    params: Any
    opt_state: Any


def _get_box_kwargs_if_npt(state):
    kwargs = {}
    if is_npt_ensemble(state):
        box = simulate.npt_box(state)
        kwargs['box'] = box
    return kwargs


def neighbor_update(neighbors, state):
    """Update neighbor lists irrespective of the ensemble.

    Fetches the box to the neighbor list update function in case of the
    NPT ensemble.

    Args:
        neighbors: Neighbor list to be updated
        state: Simulation state

    Returns:
        Updated neighbor list
    """
    kwargs = _get_box_kwargs_if_npt(state)
    nbrs = neighbors.update(state.position, **kwargs)
    return nbrs


def neighbor_allocate(neighbor_fn, state, extra_capacity=0):
    """Re-allocates neighbor lost irrespective of ensemble. Not jitable.

    Args:
        neighbor_fn: Neighbor function to re-allocate neighbor list
        state: Simulation state
        extra_capacity: Additional capacity of new neighbor list

    Returns:
        Updated neighbor list
    """
    kwargs = _get_box_kwargs_if_npt(state)
    nbrs = neighbor_fn.allocate(state.position, extra_capacity, **kwargs)
    return nbrs


def is_npt_ensemble(state):
    """Whether a state belongs to the NPT ensemble."""
    return hasattr(state, 'box_position')


def tree_combine(tree):
    """Combines the first two axes of `tree`, e.g. after batching."""
    return tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), tree)


def tree_norm(tree):
    """Returns the Euclidean norm of a PyTree."""
    leaves, _ = tree_flatten(tree)
    return sum(jnp.vdot(x, x) for x in leaves)


def tree_get_single(tree, n=0):
    """Returns the n-th tree of a tree-replica, e.g. from pmap.
    By default, the first tree is returned.
    """
    single_tree = tree_map(lambda x: jnp.array(x[n]), tree)
    return single_tree


def tree_set(tree, new_data, end, start=0):
    """Overrides entries of a tree from index start:end along axis 0
    with new_data.
    """
    return tree_map(lambda leaf, new_data_leaf:
                    leaf.at[start:end, ...].set(new_data_leaf), tree, new_data)


def tree_get_slice(tree, idx_start, idx_stop, take_every=1, to_device=True):
    """Returns a slice of trees taken from a tree-replica along axis 0."""
    if to_device:
        return tree_map(lambda x: jnp.array(x[idx_start:idx_stop:take_every]),
                        tree)
    else:
        return tree_map(lambda x: x[idx_start:idx_stop:take_every], tree)


def tree_take(tree, indicies, axis=0, on_cpu=True):
    """Tree-wise application of numpy.take."""
    numpy = onp if on_cpu else jnp
    return tree_map(lambda x: numpy.take(x, indicies, axis), tree)


def tree_replicate(tree):
    """Replicates a pytree along the first axis for pmap."""
    return tree_map(lambda x: jnp.array([x] * device_count()), tree)


def tree_concat(tree):
    """For output computed in parallel via pmap, restacks all leaves such that
    the parallel dimension is again along axis 0 and the leading pmap dimension
    vanishes.
    """
    return tree_map(partial(jnp.concatenate, axis=0), tree)


def tree_pmap_split(tree, n_devices):
    """Splits the first axis of `tree` evenly across the number of devices for
     pmap batching (size of first axis is n_devices).
     """
    assert tree_leaves(tree)[0].shape[0] % n_devices == 0, \
        'First dimension needs to be multiple of number of devices.'
    return tree_map(lambda x: jnp.reshape(x, (n_devices, x.shape[0]//n_devices,
                                              *x.shape[1:])), tree)


def tree_vmap_split(tree, batch_size):
    """Splits the first axis of a 'tree' with leaf sizes (N, X)`into
    (n_batches, batch_size, X) to allow straightforward vmapping over axis0.
    """
    assert tree_leaves(tree)[0].shape[0] % batch_size == 0, \
        'First dimension of tree needs to be splittable by batch_size' \
        ' without remainder.'
    return tree_map(lambda x: jnp.reshape(x, (x.shape[0] // batch_size,
                                              batch_size, *x.shape[1:])),
                    tree)


def tree_sum(tree_list, axis=None):
    """Computes the sum of equal-shaped leafs of a pytree."""
    @partial(partial, tree_map)
    def leaf_add(leafs):
        return jnp.sum(leafs, axis=axis)
    return leaf_add(tree_list)


def tree_mean(tree_list):
    """Computes the mean a list of equal-shaped pytrees."""
    @partial(partial, tree_map)
    def tree_add_imp(*leafs):
        return jnp.mean(jnp.stack(leafs), axis=0)

    return tree_add_imp(*tree_list)


def tree_multiplicity(tree):
    """Returns the number of stacked trees along axis 0."""
    leaves, _ = tree_flatten(tree)
    return leaves[0].shape[0]


def load_trainer(file_path):
    """Returns the trainer saved via 'trainer.save_trainer'.

    Args:
        file_path: Path of pickle file containing trainer.

    """
    with open(file_path, 'rb') as pickle_file:
        trainer = pickle.load(pickle_file)
    trainer.move_to_device()
    return trainer


def format_not_recognized_error(file_format):
    raise ValueError(f'File format {file_format} not recognized. '
                     f'Expected ".hdf5" or ".pkl".')


class TrainerInterface(abc.ABC):
    """Abstract class defining the user interface of trainers as well as
    checkpointing functionality.
    """
    def __init__(self, checkpoint_path, reference_energy_fn_template=None):
        """A reference energy_fn_template can be provided, but is not mandatory
        due to the dependence of the template on the box via the displacement
        function.
        """
        self.checkpoint_path = checkpoint_path
        self._epoch = 0
        self.reference_energy_fn_template = reference_energy_fn_template

    @property
    def energy_fn(self):
        if self.reference_energy_fn_template is None:
            raise ValueError('Cannot construct energy_fn as no reference '
                             'energy_fn_template was provided during '
                             'initialization.')
        return self.reference_energy_fn_template(self.params)

    def _dump_checkpoint_occasionally(self, frequency=None):
        """Dumps a checkpoint during training, from which training can
        be resumed.
        """
        assert self.checkpoint_path is not None
        if frequency is not None:
            pathlib.Path(self.checkpoint_path).mkdir(parents=True,
                                                     exist_ok=True)
            if self._epoch % frequency == 0:  # checkpoint model
                file_path = (self.checkpoint_path +
                             f'/epoch{self._epoch - 1}.pkl')
                self.save_trainer(file_path)

    def save_trainer(self, save_path):
        """Saves whole trainer, e.g. for production after training."""
        with open(save_path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def save_energy_params(self, file_path, save_format='.hdf5'):
        if save_format == '.hdf5':
            raise NotImplementedError
            # from jax_sgmc.io import pytree_dict_keys, dict_to_pytree
            # leaf_names = pytree_dict_keys(self.state)
            # leafes = tree_leaves(self.state)
            # with h5py.File(file_path, "w") as file:
            #     for leaf_name, value in zip(leaf_names, leafes):
            #         file[leaf_name] = value
        elif save_format == '.pkl':
            with open(file_path, 'wb') as pickle_file:
                pickle.dump(self.params, pickle_file)
        else:
            format_not_recognized_error(save_format)

    def load_energy_params(self, file_path):
        if file_path.endswith('.hdf5'):
            raise NotImplementedError
        elif file_path.endswith('.pkl'):
            with open(file_path, 'rb') as pickle_file:
                params = pickle.load(pickle_file)
        else:
            format_not_recognized_error(file_path[-4:])
        self.params = tree_map(jnp.array, params)  # move state on device

    @property
    @abc.abstractmethod
    def params(self):
        """Short-cut for parameters. Depends on specific trainer."""

    @params.setter
    @abc.abstractmethod
    def params(self, loaded_params):
        raise NotImplementedError()

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        """Training of any trainer should start by calling train."""

    @abc.abstractmethod
    def move_to_device(self):
        """Move all attributes that are expected to be on device to device to
         avoid TracerExceptions after loading trainers from disk, i.e.
         loading numpy rather than device arrays.
         """