"""Molecular dynamics observable functions acting on trajectories rather than
single snapshots.

Builds on the TrajectoryState object defined in traj_util.py.
"""
from jax import numpy as jnp


def init_traj_mean_fn(quantity_key):
    """Initializes the 'traj_fn' for the DiffTRe 'target' dict for simple
    trajectory-averaged observables.

    This function builds the 'traj_fn' of the DiffTRe 'target' dict for the
    common case of target observables that simply consist of a
    trajectory-average of instantaneous quantities, such as RDF, ADF, pressure
    or density.

    This function also serves as a template on how to build the 'traj_fn' for
    observables that are a general function of one or many instantaneous
    quantities, such as stiffness via the stress-fluctuation method or
    fluctuation formulas in this module. The 'traj_fn' receives a dict of all
    quantity trajectories as input under the same keys as instantaneous
    quantities are defined in 'quantities'. The 'traj_fn' then returns the
    ensemble-averaged quantity, possibly taking advantage of fluctuation
    formulas defined in the traj_quantity module.

    Args:
        quantity_key: Quantity key used in 'quantities' to generate the
                      quantity trajectory at hand, to be averaged over.

    Returns:
        The 'traj_fn' to be used in building the 'targets' dict for DiffTRe.
    """
    def traj_mean(quantity_trajs):
        quantity_traj = quantity_trajs[quantity_key]
        return jnp.mean(quantity_traj, axis=0)
    return traj_mean
