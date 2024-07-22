import sympy as sym
import numpy as np
from sklearn.preprocessing import PolynomialFeatures



def differentiate_model(polynomials, coefficients, x):
    """
    Differentiate a model with polynomial features.

    Parameters:
    polynomials (PolynomialFeatures): Polynomial features object.
    coefficients (np.ndarray): Coefficients of the model.
    x (np.ndarray) of shape (n_features, n_samples): Input data.

    Returns:
    np.ndarray: Derivative of the model.
    """
    x = x.T # to make it compatible with sklearn
    derivative = []
    variables = x.shape[1]
    for poww in polynomials.powers_:
        derivative_entries = []
        for i in range(variables):
            entry = 1
            for j, p in enumerate(poww):
                if i == j: 
                    entry *= p * x[:, j]**(p-1)
                else:
                    entry *= x[:, j]**p
            derivative_entries.append(entry)
        derivative.append(derivative_entries)
    nonlinear_features_of_derivative = np.array(derivative).transpose(2, 0, 1) # shape (n_samples, n_features_nonlin, n_features)
    return np.matmul(coefficients, nonlinear_features_of_derivative).transpose(1, 2, 0) # shape (n_features, n_features, n_samples)

def differentiate_model_symbolic(polynomials, coefficients):
    """
    Differentiate a model with polynomial features using symbolic computation.

    Parameters:
    polynomials (PolynomialFeatures): Polynomial features object.
    coefficients (np.ndarray): Coefficients of the model.

    Returns:
    tuple: Symbolic matrix of transformation, Jacobian, and the variables.
    """
        
    powers = polynomials.powers_.T
    n_variables = powers.shape[0]
    base_symbol = 'x'
    variables = [sym.symbols('%s_%d' %(base_symbol, i)) for i in range(n_variables)]
    n_equations = coefficients.shape[0]
    equations = []
    for k in range(n_equations):
        term = 0
        for l,p in enumerate(powers.T): 
            prod = 1
            for i, p_ in enumerate(p):
                prod *= variables[i]**p_
            term += coefficients[k,l]*prod

        equations.append(term)
    return sym.Matrix(equations), sym.Matrix(equations).jacobian(variables), variables
