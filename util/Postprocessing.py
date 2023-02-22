import jax.numpy as jnp
from jax import lax
import pickle

from matplotlib import pyplot as plt
from functools import partial


def box_density(R_snapshot, bin_edges, axis=0):
    # assumes all particles are wrapped into the same box
    profile, _ = jnp.histogram(R_snapshot[:, axis], bins=bin_edges)
    # norm via n_bins and n_particles
    profile *= (profile.shape[0] / R_snapshot.shape[0])
    return profile


def get_bin_centers_from_edges(bin_edges):
    """To get centers from bin edges as generated from jnp.histogram"""
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.
    return bin_centers


def plot_density(file_name, n_bins=50):
    with open(file_name, 'rb') as f:
        R_traj_list, box = pickle.load(f)

    R_traj = R_traj_list[10]
    bin_edges = jnp.linspace(0., box[0], n_bins + 1)
    bin_centers = get_bin_centers_from_edges(bin_edges)
    compute_box_density = partial(box_density, bin_edges=bin_edges)
    density_snapshots = lax.map(compute_box_density, R_traj)
    density = jnp.mean(density_snapshots, axis=0)

    file_name = file_name[:-4]
    plt.figure()
    plt.plot(bin_centers, density)
    plt.ylabel('Normalizes Density')
    plt.xlabel('x')
    plt.savefig(file_name + '.png')


def plot_initial_and_predicted_rdf(rdf_bin_centers, g_average_final, model,
                                   visible_device, reference_rdf=None,
                                   g_average_init=None, after_pretraining=False,
                                   std=None, T=None, color=None):
    if color is None:
        color = ['k', '#00A087FF', '#3C5488FF']
    if after_pretraining:
        pretrain_str = '_after_pretrain'
    else:
        pretrain_str = ''

    plt.figure()
    plt.plot(rdf_bin_centers, g_average_final, label='predicted',
             color=color[1])
    if g_average_init is not None:
        plt.plot(rdf_bin_centers, g_average_init,  label='initial guess',
                 color=color[2])
    if reference_rdf is not None:
        plt.plot(rdf_bin_centers, reference_rdf, label='reference',
                 dashes=(4, 3), color=color[0], linestyle='--')
    if std is not None:
        plt.fill_between(rdf_bin_centers, g_average_final - std,
                         g_average_final + std, alpha=0.3,
                         facecolor='#00A087FF', label='Uncertainty')
    plt.legend()
    plt.xlabel('r in $\mathrm{nm}$')
    plt.savefig(f'output/figures/predicted_RDF_'
                f'{model}_{T or ""}_{visible_device}{pretrain_str}.png')


def plot_initial_and_predicted_adf(adf_bin_centers, predicted_adf_final, model,
                                   visible_device, reference_adf=None,
                                   adf_init=None, after_pretraining=False,
                                   std=None, T=None, color=None):
    if color is None:
        color = ['k', '#00A087FF', '#3C5488FF']
    if after_pretraining:
        pretrain_str = '_after_pretrain'
    else:
        pretrain_str = ''

    plt.figure()
    plt.plot(adf_bin_centers, predicted_adf_final, label='predicted',
             color=color[1])
    if adf_init is not None:
        plt.plot(adf_bin_centers, adf_init, label='initial guess',
                 color=color[2])
    if reference_adf is not None:
        plt.plot(adf_bin_centers, reference_adf, label='reference',
                 dashes=(4, 3), color=color[0], linestyle='--')
    if std is not None:
        plt.fill_between(adf_bin_centers, predicted_adf_final - std,
                         predicted_adf_final + std, alpha=0.3,
                         facecolor='#00A087FF', label='Uncertainty')
    plt.legend()
    plt.xlabel(r'$\alpha$ in $\mathrm{rad}$')
    plt.savefig(f'output/figures/predicted_ADF_'
                f'{model}_{T or ""}_{visible_device}{pretrain_str}.png')


def plot_initial_and_predicted_tcf(bin_centers, g_average_final, model,
                                   visible_device, reference_tcf=None,
                                   tcf_init=None, labels=None,
                                   axis_label=None, color=None):
    if color is None:
        color = ['k', '#00A087FF', '#3C5488FF']
    if labels is None:
        labels = ['reference', 'predicted', 'initial guess']
    plt.figure()
    plt.plot(bin_centers, g_average_final, label=labels[1], color=color[1])
    if tcf_init is not None:
        plt.plot(bin_centers, tcf_init, label=labels[2], color=color[2])
    if reference_tcf is not None:
        plt.plot(bin_centers, reference_tcf, label=labels[0], dashes=(4, 3),
                 color=color[0], linestyle='--')
    plt.legend()
    if axis_label is not None:
        plt.ylabel(axis_label[0])
        plt.xlabel(axis_label[1])
    else:
        plt.xlabel('r in $\mathrm{nm}$')
    
    plt.savefig(f'output/figures/predicted_TCF_'
                f'{model}_{visible_device}.png')
