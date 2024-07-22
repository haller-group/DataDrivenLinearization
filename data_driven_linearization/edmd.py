import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def edmd(x, degree=3):
    """
    EDMD model with no regularization and polynomial features.

    Parameters:
    x (list of np.ndarray): List of state matrices.
    degree (int): Degree of polynomial features.

    Returns:
    LinearRegression: Fitted EDMD model.
    """
    nonlinear_features = [PolynomialFeatures(degree=degree, include_bias=False).fit_transform(x_element.T) for x_element in x]
    XX_EDMD = [nonlinear_feature_element[:-1, :] for nonlinear_feature_element in nonlinear_features]
    YY_EDMD = [nonlinear_feature_element[1:, :] for nonlinear_feature_element in nonlinear_features]
    XX_EDMD = np.concatenate(XX_EDMD)
    YY_EDMD = np.concatenate(YY_EDMD)
    edmd_model = LinearRegression(fit_intercept=False)
    edmd_model.fit(XX_EDMD, YY_EDMD)
    return edmd_model
