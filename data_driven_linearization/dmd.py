import numpy as np
from sklearn.linear_model import LinearRegression


def dmd(x):
    """
    DMD model with no regularization.

    Parameters:
    x (list of np.ndarray): List of state matrices.

    Returns:
    LinearRegression: Fitted DMD model.
    """
    XX = [x_element[:, :-1].T for x_element in x]
    YY = [x_element[:, 1:].T for x_element in x]
    XX = np.concatenate(XX)
    YY = np.concatenate(YY)
    dmd_model = LinearRegression(fit_intercept=False)
    dmd_model.fit(XX, YY)
    return dmd_model

