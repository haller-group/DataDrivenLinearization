import numpy as np
import logging

logger = logging.getLogger("coordinates_embedding")

def coordinates_embedding(t, x, imdim=None, over_embedding=0, force_embedding=False, time_stepping=1, shift_steps=1):
    """
    Returns the n-dim. time series x into a time series of properly embedded
    coordinate system y of dimension p.

    Parameters:
    t (list of np.ndarray): List of time vectors.
    x (list of np.ndarray): List of observed trajectories.
    imdim (int): Dimension of the invariant manifold to learn.
    over_embedding (int): Augment the minimal embedding dimension with time-delayed measurements.
    force_embedding (bool): Force the embedding in the states of x.
    time_stepping (int): Time stepping in the time series.
    shift_steps (int): Number of timesteps passed between components.

    Returns:
    tuple: List of time vectors, embedded trajectories, and embedding options.
    """
    if not imdim:
        raise RuntimeError("imdim not specified for coordinates embedding")
    n_observables = x[0].shape[0]
    n_n = int(np.ceil((2*imdim + 1)/n_observables) + over_embedding)

    if n_n > 1 and not force_embedding:
        p = n_n * n_observables
        if n_observables == 1:
            logger.info(f'The {p} embedding coordinates consist of the measured state and its {n_n-1} time-delayed measurements.')
        else:
            logger.info(f'The {p} embedding coordinates consist of the {n_observables} measured states and their {n_n-1} time-delayed measurements.')
        t_y, y = [], []
        for i_traj in range(len(x)):
            t_i = t[i_traj]
            x_i = x[i_traj]
            subsample = np.arange(start=0, stop=len(t_i), step=time_stepping)
            y_i = x_i[:, subsample]
            y_base = x_i[:, subsample]
            for i_rep in range(1, n_n):
                y_i = np.concatenate((y_i, np.roll(y_base, -i_rep)))
            y.append(y_i[:, :-n_n+1])
            t_y.append(t_i[subsample[:-n_n+1]])
    else:
        p = n_observables
        if time_stepping > 1:
            logger.info('The embedding coordinates consist of the measured states.')
            t_y, y = [], []
            for i_traj in range(len(x)):
                t_i = t[i_traj]
                x_i = x[i_traj]
                subsample = np.arange(start=0, stop=len(t_i), step=time_stepping)
                t_y.append(t_i[subsample])
                y.append(x_i[:, subsample])
        else:
            t_y, y = t, x

    opts_embedding = {
        'imdim': imdim,
        'over_embedding': over_embedding,
        'force_embedding': force_embedding,
        'time_stepping': time_stepping,
        'shift_steps': shift_steps,
        'embedding_space_dim': p
    }

    return t_y, y, opts_embedding
