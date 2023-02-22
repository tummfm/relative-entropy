"""Functions to extract the sparse directional graph representation of a
molecular state.

The :class:`SparseDirectionalGraph` is the input to
:class:`~chemtrain.neural_networks.DimeNetPP`.
"""
import inspect
from typing import Optional, Callable, Tuple

import chex
from jax import numpy as jnp, vmap, lax
from jax_md import space, partition, smap


@chex.dataclass
class SparseDirectionalGraph:
    """Sparse directial graph representation of a molecular state.

     Required arguments are necessary inputs for DimeNet++.
     If masks are not provided, all entities are assumed to be present.

     Attributes:
         distance_ij: A (N_edges,) array storing for each the radial distances
                      between particle i and j
         idx_i: A (N_edges,) array storing for each edge particle index i
         idx_j: A (N_edges,) array storing for each edge particle index j
         angles: A (N_triplets,) array storing for each triplet the angle formed
                 by the 3 particles
         reduce_to_ji: A (N_triplets,) array storing for each triplet kji edge
                       index j->i to aggregate messages via a segment_sum: each
                       m_ji is a distinct segment containing all incoming m_kj.
         expand_to_kj: A (N_triplets,) array storing for each triplet kji edge
                       index k->j to gather all incoming edges for message
                       passing.
         edge_mask: A (N_edges,) boolean array storing for each edge whether the
                    edge exists. By default, all edges are considered.
         triplet_mask: A (N_triplets,) boolean array storing for each triplet
                       whether the triplet exists. By default, all triplets are
                       considered.
         n_edges: Number of non-masked edges in the graph. None assumes all
                  edges are real.
         n_triplets: Number of non-masked triplets in the graph. None assumes
                     all triplets are real.
         n_particles: Number of non-masked species in the graph.
    """
    species: jnp.ndarray
    distance_ij: jnp.ndarray
    idx_i: jnp.ndarray
    idx_j: jnp.ndarray
    angles: jnp.ndarray
    reduce_to_ji: jnp.ndarray
    expand_to_kj: jnp.ndarray
    species_mask: Optional[jnp.ndarray] = None
    edge_mask: Optional[jnp.ndarray] = None
    triplet_mask: Optional[jnp.ndarray] = None
    n_edges: Optional[int] = None
    n_triplets: Optional[int] = None

    def __post_init__(self):
        if self.species_mask is None:
            self.species_mask = jnp.ones_like(self.species, dtype=bool)
        if self.edge_mask is None:
            self.edge_mask = jnp.ones_like(self.distance_ij, dtype=bool)
        if self.triplet_mask is None:
            self.triplet_mask = jnp.ones_like(self.angles, dtype=bool)

    @property
    def n_particles(self):
        return jnp.sum(self.species_mask)

    def to_dict(self):
        """Returns the stored graph data as a dictionary of arrays.
        This format is often beneficial for dataloaders.
        """
        return {
            'species': self.species,
            'distance_ij': self.distance_ij,
            'idx_i': self.idx_i,
            'idx_j': self.idx_j,
            'angles': self.angles,
            'reduce_to_ji': self.reduce_to_ji,
            'expand_to_kj': self.expand_to_kj,
            'species_mask': self.species_mask,
            'edge_mask': self.edge_mask,
            'triplet_mask': self.triplet_mask
        }

    @classmethod
    def from_dict(cls, graph_dict):
        """Initializes instance from dictionary containing all necessary keys
        for initialization.
        """
        return cls(**{
            key: value for key, value in graph_dict.items()
            if key in inspect.signature(cls).parameters
        })

    def cap_exactly(self):
        """Deletes all non-existing edges and triplets from the stored graph.

        This is a non-pure function and hence not available in a jit-context.
        Returning the capped graph does not solve the problem when n_edges
        and n_triplets are computed within the jit-compiled function.
        """
        # edges are sorted, hence all non-existing edges are at the end
        self.species = self.species[:self.n_particles]
        self.species_mask = self.species_mask[:self.n_particles]

        self.distance_ij = self.distance_ij[:self.n_edges]
        self.idx_i = self.idx_i[:self.n_edges]
        self.idx_j = self.idx_j[:self.n_edges]
        self.edge_mask = self.edge_mask[:self.n_edges]

        self.angles = self.angles[:self.n_triplets]
        self.reduce_to_ji = self.reduce_to_ji[:self.n_triplets]
        self.expand_to_kj = self.expand_to_kj[:self.n_triplets]
        self.triplet_mask = self.triplet_mask[:self.n_triplets]


def angle(r_ij, r_kj):
    """Computes the angle (kj, ij) from vectors r_kj and r_ij,
    correctly selecting the quadrant.

    Based on
    :math:`\\tan(\\theta) = |(r_{ji} \\times r_{kj})| / (r_{ji} \\cdot r_{kj})`.
    Beware the non-differentability of arctan2(0,0).

    Args:
        r_ij: Vector pointing to position of particle i from particle j
        r_kj: Vector pointing to position of particle k from particle j

    Returns:
        Angle between vectors
    """
    cross = jnp.linalg.norm(jnp.cross(r_ij, r_kj))
    dot = jnp.dot(r_ij, r_kj)
    theta = jnp.arctan2(cross, dot)
    return theta


def safe_angle_mask(r_ji, r_kj, angle_mask):
    """Sets masked angles to pi/2 to ensure differentiablility.

    Args:
        r_ji: Array (N_triplets, dim) of vectors  pointing to position of
              particle i from particle j
        r_kj: Array (N_triplets, dim) of vectors pointing to position of
              particle k from particle j
        angle_mask: (N_triplets, ) or (N_triplets, 1) Boolean mask for each
                    triplet, which is False for triplets that need to be masked.

    Returns:
        A tuple (r_ji_safe, r_kj_safe) of vectors r_ji and r_kj, where masked
        triplets are replaced such that the angle between them is pi/2.
    """
    if angle_mask.ndim == 1:  # expand for broadcasing, if necessary
        angle_mask = jnp.expand_dims(angle_mask, -1)
    safe_ji = jnp.array([1., 0., 0.], dtype=jnp.float32)
    safe_kj = jnp.array([0., 1., 0.], dtype=jnp.float32)
    r_ji_safe = jnp.where(angle_mask, r_ji, safe_ji)
    r_kj_safe = jnp.where(angle_mask, r_kj, safe_kj)
    return r_ji_safe, r_kj_safe


def angle_triplets(positions, displacement_fn, angle_idxs, angle_mask):
    """Computes the angle for all triplets between 0 and pi.

     Masked angles are set to pi/2.

    Args:
        positions: Array pf particle positions (N_particles x 3)
        displacement_fn: Jax_md displacement function
        angle_idxs: Array of particle indeces that form a triplet
                    (N_triples x 3)
        angle_mask: Boolean mask for each triplet, which is False for triplets
                    that need to be masked.

    Returns:
        A (N_triples,) array with the angle for each triplet.
    """
    r_i = positions[angle_idxs[:, 0]]
    r_j = positions[angle_idxs[:, 1]]
    r_k = positions[angle_idxs[:, 2]]

    # Note: The original DimeNet implementation uses R_ji, however r_ij is the
    #       correct vector to get the angle between both vectors. This is a
    #       known issue in DimeNet. We apply the correct angle definition.
    r_ij = vmap(displacement_fn)(r_i, r_j)  # r_i - r_j respecting periodic BCs
    r_kj = vmap(displacement_fn)(r_k, r_j)
    # we need to mask as the case where r_ij is co-linear with r_kj.
    # Otherwise, this generates NaNs on the backward pass
    r_ij_safe, r_kj_safe = safe_angle_mask(r_ij, r_kj, angle_mask)
    angles = vmap(angle)(r_ij_safe, r_kj_safe)
    return angles


def _flatten_sort_and_capp(matrix, sorting_args, cap_size):
    """Helper function that takes a 2D array, flattens it, sorts it using the
    args (usually provided via argsort) and capps the end of the resulting
    vector. Used to delete non-existing edges and returns the capped vector.
    """
    vect = jnp.ravel(matrix)
    sorted_vect = vect[sorting_args]
    capped_vect = sorted_vect[0:cap_size]
    return capped_vect


def sparse_graph_from_neighborlist(displacement_fn: Callable,
                                   positions: jnp.ndarray,
                                   neighbor: partition.NeighborList,
                                   r_cutoff: jnp.array,
                                   species: jnp.array = None,
                                   max_edges: Optional[int] = None,
                                   max_triplets: Optional[int] = None,
                                   species_mask: jnp.array = None,
                                   ) -> Tuple[SparseDirectionalGraph, bool]:
    """Constructs a sparse representation of graph edges and angles to save
    memory and computations over neighbor list.

    The speed-up over simply using the dense jax_md neighbor list is
    significant, particularly regarding triplets. To allow for a representation
    of constant size required by jit, we pad the resulting vectors.

    Args:
        displacement_fn: Jax_MD displacement function encoding box dimensions
        positions: (N_particles, dim) array of particle positions
        neighbor: Jax_MD neighbor list that is in sync with positions
        r_cutoff: Radial cutoff distance, below which 2 particles are considered
                  to be connected by an edge.
        species: (N_particles,) array encoding atom types. If None, assumes type
                 0 for all atoms.
        max_edges: Maximum number of edges storable in the graph. Can be used to
                   reduce the number of padded edges, but should be used
                   carefully, such that no existing edges are capped. Default
                   None uses the maximum possible number of edges as given by
                   the dense neighbor list.
        max_triplets: Maximum number of triplets storable in the graph. Can be
                    used to reduce the number of padded triplets, but should be
                    used carefully, such that no existing triplets are capped.
                    Default None uses the maximum possible number of triplets as
                    given by the dense neighbor list.
        species_mask: (N_particles,) array encoding atom types. Default None,
                    assumes no masking necessary.

    Returns:
        Tuple (sparse_graph, too_many_edges_error_code) containing the
        SparseDirectionalGraph and whether max_edges or max_triplets overflowed.
    """
    assert neighbor.format.name == 'Dense', ('Currently only dense neighbor'
                                             ' lists supported.')
    n_particles, max_neighbors = neighbor.idx.shape
    species = _canonicalize_species(species, n_particles)

    neighbor_displacement_fn = space.map_neighbor(displacement_fn)

    # compute pairwise distances
    pos_neigh = positions[neighbor.idx]
    pair_displacement = neighbor_displacement_fn(positions, pos_neigh)
    pair_distances = space.distance(pair_displacement)

    # compute adjacency matrix via neighbor_list, then build sparse graph
    # representation to avoid part of padding overhead in dense neighborlist
    # adds all edges > cut-off to masked edges
    edge_idx_ji = jnp.where(pair_distances < r_cutoff, neighbor.idx,
                            n_particles)
    # neighbor.idx: an index j in row i encodes a directed edge from
    #               particle j to particle i.
    # edge_idx[i, j]: j->i. if j == N: encodes masked edge.
    # Index N would index out-of-bounds, but in jax the last element is
    # returned instead

    # conservative estimates for initialization run
    # use guess from initialization for tighter bound to save memory and
    # computations during production runs
    if max_edges is None:
        max_edges = n_particles * max_neighbors
    if max_triplets is None:
        max_triplets = max_edges * max_neighbors

    # sparse edge representation:
    # construct vectors from adjacency matrix and only keep existing edges
    # Target node (i) and source (j) of edges
    pair_mask = edge_idx_ji != n_particles  # non-existing neighbor encoded as N
    # due to undirectedness, each edge is included twice
    n_edges = jnp.count_nonzero(pair_mask)
    pair_mask_flat = jnp.ravel(pair_mask)
    # non-existing edges are sorted to the end for capping
    sorting_idxs = jnp.argsort(~pair_mask_flat)
    _, yy = jnp.meshgrid(jnp.arange(max_neighbors), jnp.arange(n_particles))  # pylint: disable=unbalanced-tuple-unpacking
    idx_i = _flatten_sort_and_capp(yy, sorting_idxs, max_edges)
    idx_j = _flatten_sort_and_capp(edge_idx_ji, sorting_idxs, max_edges)
    d_ij = _flatten_sort_and_capp(pair_distances, sorting_idxs, max_edges)
    sparse_pair_mask = _flatten_sort_and_capp(pair_mask_flat, sorting_idxs,
                                              max_edges)

    # build sparse angle combinations from adjacency matrix:
    # angle defined for 3 particles with connections k->j and j->i
    # directional message passing accumulates all k->j to update each m_ji
    idx3_i = jnp.repeat(idx_i, max_neighbors)
    idx3_j = jnp.repeat(idx_j, max_neighbors)
    # retrieves for each j in idx_j its neighbors k: stored in 2nd axis
    idx3_k_mat = edge_idx_ji[idx_j]
    idx3_k = idx3_k_mat.ravel()
    angle_idxs = jnp.column_stack([idx3_i, idx3_j, idx3_k])

    # masking:
    # k and j are different particles, by edge_idx_ji construction.
    # The same applies to j - i, except for masked ones
    mask_i_eq_k = idx3_i != idx3_k
    # mask for ij known a priori
    mask_ij = jnp.repeat(sparse_pair_mask, max_neighbors)
    mask_k = idx3_k != n_particles
    angle_mask = mask_ij * mask_k * mask_i_eq_k  # union of masks
    angle_mask, sorting_idx3 = lax.top_k(angle_mask, max_triplets)
    angle_idxs = angle_idxs[sorting_idx3]
    n_triplets = jnp.count_nonzero(angle_mask)
    angles = angle_triplets(positions, displacement_fn, angle_idxs, angle_mask)

    # retrieving edge_id m_ji from nodes i and j:
    # idx_i < N by construction, but idx_j can be N: will override
    # lookup[i, N-1], which is problematic if [i, N-1] is an existing edge.
    # Hence, the lookup table is extended by 1.
    edge_id_lookup = jnp.zeros([n_particles, n_particles + 1], dtype=jnp.int32)
    edge_id_lookup_direct = edge_id_lookup.at[(idx_i, idx_j)].set(
        jnp.arange(max_edges))

    # stores for each angle kji edge index j->i to aggregate messages via a
    # segment_sum: each m_ji is a distinct segment containing all incoming m_kj
    reduce_to_ji = edge_id_lookup_direct[(angle_idxs[:, 0], angle_idxs[:, 1])]
    # stores for each angle kji edge index k->j to gather all incoming edges
    # for message passing
    expand_to_kj = edge_id_lookup_direct[(angle_idxs[:, 1], angle_idxs[:, 2])]

    too_many_edges_error_code = lax.cond(
        jnp.bitwise_or(n_edges > max_edges, n_triplets > max_triplets),
        lambda _: True, lambda _: False, n_edges
    )

    sparse_graph = SparseDirectionalGraph(
        species=species, distance_ij=d_ij, idx_i=idx_i, idx_j=idx_j,
        angles=angles, reduce_to_ji=reduce_to_ji, expand_to_kj=expand_to_kj,
        edge_mask=sparse_pair_mask, triplet_mask=angle_mask, n_edges=n_edges,
        n_triplets=n_triplets, species_mask=species_mask
    )
    return sparse_graph, too_many_edges_error_code


def _canonicalize_species(species, n_particles):
    """Ensures species are integer and initializes species to 0 if species=None.

    Args:
        species: (N_particles,) array of atom types or None
        n_particles: Number of particles

    Returns:
        Integer species array.
    """
    if species is None:
        species = jnp.zeros(n_particles, dtype=jnp.int32)
    else:
        smap._check_species_dtype(species)  # assert species are int
    return species