"""Custom definition of some potential energy functions."""
from functools import partial
from typing import Callable, Any

from jax import vmap
import jax.numpy as jnp
from jax_md import space, partition, util, energy, smap

from chemtrain.jax_md_mod import custom_interpolate, custom_quantity
from chemtrain import sparse_graph

# Types
f32 = util.f32
f64 = util.f64
Array = util.Array

PyTree = Any
Box = space.Box
DisplacementFn = space.DisplacementFn
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList


def harmonic_angle(displacement_or_metric: DisplacementOrMetricFn,
                   angle_idxs: Array,
                   eq_mean: Array,
                   eq_variance: Array,
                   kbt: [float, Array]):
    """Harmonic Angle interaction.

    The variance of the angle is used to determine the force constant.
    https://manual.gromacs.org/documentation/2019/reference-manual/functions/bonded-interactions.html

    Args:
        displacement_or_metric: Displacement function
        angle_idxs: Indices of particles (i, j, k)
        eq_mean: Equilibrium angle in degrees
        eq_variance: Angle Variance
        kbt: kbT

    Returns:
        Harmonic angle potential energy function.
    """

    kbt = jnp.array(kbt, dtype=f32)
    angle_mask = jnp.ones([angle_idxs.shape[0], 1])
    harmonic_fn = partial(energy.simple_spring, length=eq_mean,
                          epsilon=kbt / eq_variance)

    def energy_fn(pos, **unused_kwargs):
        angles = sparse_graph.angle_triplets(pos, displacement_or_metric,
                                             angle_idxs, angle_mask)
        return jnp.sum(harmonic_fn(jnp.rad2deg(angles)))

    return energy_fn


def dihedral_energy(angle,
                    phase_angle: Array,
                    force_constant: Array,
                    n: [int, Array]):
    """Energy of dihedral angles.

    https://manual.gromacs.org/documentation/2019/reference-manual/functions/bonded-interactions.html
    """
    cos_angle = jnp.cos(n * angle - phase_angle)
    energies = force_constant * (1 + cos_angle)
    return jnp.sum(energies)


def periodic_dihedral(displacement_or_metric: DisplacementOrMetricFn,
                      dihedral_idxs: Array,
                      phase_angle: Array,
                      force_constant: Array,
                      multiplicity: [float, Array]):
    """Peridoc dihedral angle interaction.

    https://manual.gromacs.org/documentation/2019/reference-manual/functions/bonded-interactions.html

    Args:
        displacement_or_metric: Displacement function
        dihedral_idxs: Indices of particles (i, j, k, l) building the dihedrals
        phase_angle: Dihedral phase angle in degrees.
        force_constant: Force constant
        multiplicity: Dihedral multiplicity

    Returns:
        Peridoc dihedral potential energy function.
    """

    multiplicity = jnp.array(multiplicity, dtype=f32)
    phase_angle = jnp.deg2rad(phase_angle)

    def energy_fn(pos, **unused_kwargs):
        dihedral_angles = custom_quantity.dihedral_displacement(
            pos, displacement_or_metric, dihedral_idxs, degrees=False)
        per_angle_u = vmap(dihedral_energy)(dihedral_angles, phase_angle,
                                            force_constant, multiplicity)
        return jnp.sum(per_angle_u)

    return energy_fn


def generic_repulsion(dr: Array,
                      sigma: Array = 1.,
                      epsilon: Array = 1.,
                      exp: Array = 12.,
                      **unused_dynamic_kwargs) -> Array:
    """
    Repulsive interaction between soft sphere particles:
    U = epsilon * (sigma / r)**exp.

    Args:
      dr: An ndarray of pairwise distances between particles.
      sigma: Repulsion length scale
      epsilon: Interaction energy scale
      exp: Exponent specifying interaction stiffness

    Returns:
      Array of energies
    """
    del unused_dynamic_kwargs
    dr = jnp.where(dr > 1.e-7, dr, 1.e7)  # save masks dividing by 0
    idr = (sigma / dr)
    pot_energy = epsilon * idr ** exp
    return pot_energy


def generic_repulsion_neighborlist(
        displacement_or_metric: DisplacementOrMetricFn,
        box_size: Box = None,
        species: Array = None,
        sigma: Array = 1.0,
        epsilon: Array = 1.0,
        exp: [int, Array] = 12.,
        r_onset: Array = 0.9,
        r_cutoff: Array = 1.,
        dr_threshold: float = 0.2,
        per_particle: bool = False,
        capacity_multiplier: float = 1.25,
        initialize_neighbor_list: bool = True):
    """Convenience wrapper to compute generic repulsion energy over a system
    with neighborlist.

    Provides option not to initialize neighborlist. This is useful if energy
    function needs to be initialized within a jitted function.
    """
    sigma = jnp.array(sigma, dtype=f32)
    epsilon = jnp.array(epsilon, dtype=f32)
    exp = jnp.array(exp, dtype=f32)
    r_onset = jnp.array(r_onset, dtype=f32)
    r_cutoff = jnp.array(r_cutoff, dtype=f32)

    energy_fn = smap.pair_neighbor_list(
      energy.multiplicative_isotropic_cutoff(generic_repulsion, r_onset,
                                             r_cutoff),
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      species=species,
      sigma=sigma,
      epsilon=epsilon,
      exp=exp,
      reduce_axis=(1,) if per_particle else None)

    if initialize_neighbor_list:
        assert box_size is not None
        neighbor_fn = partition.neighbor_list(
            displacement_or_metric, box_size, r_cutoff, dr_threshold,
            capacity_multiplier=capacity_multiplier
        )
        return neighbor_fn, energy_fn

    return energy_fn


def generic_repulsion_nonbond(displacement_or_metric: DisplacementOrMetricFn,
                              pair_idxs: Array,
                              sigma: Array = 1.,
                              epsilon: Array = 1.,
                              exp: Array = 12.) -> Callable[[Array], Array]:
    """Convenience wrapper to compute repulsive part of Lennard Jones energy of
    particles via connection idxs.

    Args:
        displacement_or_metric: Displacement_fn
        pair_idxs: Set of pair indices (i, j) defining repulsion pairs
        sigma: sigma
        epsilon: epsilon
        exp: LJ exponent

    Returns:
        Pairwise nonbonded repulsion potential energy function.
    """
    sigma = jnp.array(sigma, f32)
    epsilon = jnp.array(epsilon, f32)
    exp = jnp.array(exp, dtype=f32)

    return smap.bond(
            generic_repulsion,
            space.canonicalize_displacement_or_metric(displacement_or_metric),
            pair_idxs,
            ignore_unused_parameters=True,
            sigma=sigma,
            epsilon=epsilon,
            exp=exp)


def lennard_jones_nonbond(displacement_or_metric: DisplacementOrMetricFn,
                          pair_idxs: Array,
                          sigma: Array = 1.,
                          epsilon: Array = 1.) -> Callable[[Array], Array]:
    """Convenience wrapper to compute lennard jones energy of nonbonded
    particles.

    Args:
        displacement_or_metric: Displacement_fn
        pair_idxs: Set of pair indices (i, j) defining repulsion pairs
        sigma: sigma
        epsilon: epsilon

    Returns:
        Pairwise nonbonded repulsion potential energy function.
    """
    sigma = jnp.array(sigma, f32)
    epsilon = jnp.array(epsilon, f32)
    return smap.bond(
            energy.lennard_jones,
            space.canonicalize_displacement_or_metric(displacement_or_metric),
            pair_idxs,
            ignore_unused_parameters=True,
            sigma=sigma,
            epsilon=epsilon)


def tabulated(dr: Array, spline: Callable[[Array], Array], **unused_kwargs
              ) -> Array:
    """
    Tabulated radial potential between particles given a spline function.

    Args:
        dr: An ndarray of pairwise distances between particles
        spline: A function computing the spline values at a given pairwise
                distance.

    Returns:
        Array of energies
    """

    return spline(dr)


def tabulated_neighbor_list(displacement_or_metric: DisplacementOrMetricFn,
                            x_vals: Array,
                            y_vals: Array,
                            box_size: Box,
                            degree: int = 3,
                            monotonic: bool = True,
                            r_onset: Array = 0.9,
                            r_cutoff: Array = 1.,
                            dr_threshold: Array = 0.2,
                            species: Array = None,
                            capacity_multiplier: float = 1.25,
                            initialize_neighbor_list: bool = True,
                            per_particle: bool = False,
                            fractional=True):
    """
    Convenience wrapper to compute tabulated energy using a neighbor list.

    Provides option not to initialize neighborlist. This is useful if energy
    function needs to be initialized within a jitted function.
    """

    x_vals = jnp.array(x_vals, f32)
    y_vals = jnp.array(y_vals, f32)
    box_size = jnp.array(box_size, f32)
    r_onset = jnp.array(r_onset, f32)
    r_cutoff = jnp.array(r_cutoff, f32)
    dr_threshold = jnp.array(dr_threshold, f32)

    # Note: cannot provide the spline parameters via kwargs because only
    #       per-particle parameters are supported
    if monotonic:
        spline = custom_interpolate.MonotonicInterpolate(x_vals, y_vals)
    else:
        spline = custom_interpolate.InterpolatedUnivariateSpline(x_vals, y_vals,
                                                                 k=degree)
    tabulated_partial = partial(tabulated, spline=spline)

    energy_fn = smap.pair_neighbor_list(
      energy.multiplicative_isotropic_cutoff(tabulated_partial, r_onset,
                                             r_cutoff),
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      species=species,
      reduce_axis=(1,) if per_particle else None)

    if initialize_neighbor_list:
        neighbor_fn = partition.neighbor_list(
            displacement_or_metric, box_size, r_cutoff, dr_threshold,
            capacity_multiplier=capacity_multiplier,
            fractional_coordinates=fractional)
        return neighbor_fn, energy_fn
    return energy_fn
