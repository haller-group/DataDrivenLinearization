import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize, least_squares
import torch
from torch.autograd.functional import hessian, jacobian
import sympy as sym

from .differentiation_utils import differentiate_model, differentiate_model_symbolic
from .dmd import dmd
import warnings

warnings.filterwarnings("ignore",category=FutureWarning)


class DataDrivenLinearization:
    """
    Main class for data-driven linearization using polynomial transformations

    Attributes:
    dimension (int): Dimension of the state space.
    degree (int): Degree of polynomial features.
    poly (PolynomialFeatures): Polynomial features object.
    n_features (int): Number of polynomial features.
    linear_model (np.ndarray): Linear model coefficients.
    transformation_coefficients (np.ndarray): Transformation coefficients.
    transform_to_diagonal (np.ndarray): Transformation to diagonal matrix.
    transform_to_nondiagonal (np.ndarray): Transformation to non-diagonal matrix.
    symbolic_transformation_to_linearized (dict): Symbolic transformation to linearized coordinates.
    symbolic_transformation_to_original (dict): Symbolic transformation to original coordinates.
    """
    def __init__(self, dimension, degree = 3):
        """
        Initialize the DataDrivenLinearization class.

        Parameters:
        dimension (int): Dimension of the phase space.
        degree (int): Degree of polynomial features.
        """
        self.degree = degree
        self.poly = PolynomialFeatures(degree = degree, include_bias = False) # will assume that the origin is a fixed point
        self.dimension = dimension
        self.n_features = self.poly.fit_transform(np.ones((1, self.dimension))).shape[1]
        self.linear_model = None
        self.transformation_coefficients = None
        self.transform_to_diagonal = None
        self.transform_to_nondiagonal = None
        self.symbolic_transformation_to_linearized = {}
        self.symbolic_transformation_to_original = {}
        

    def _prepare_initial_guess(self, 
                               initial_matrix = None,
                                initial_transformation = None):
        
        """
        Prepare the initial guess for the non-convex optimization. If no initial guess is provided, random values are used. If 'zero' is provided, the initial transformation is set to zero.

        Parameters:
        initial_matrix (np.ndarray): Initial matrix for the linear model.
        initial_transformation (np.ndarray): Initial transformation matrix.

        Returns:
        np.ndarray: Concatenated initial guess vector.
        """
        if initial_matrix is None:
            initial_matrix = np.random.rand(self.dimension, self.dimension)
        if initial_transformation is None:
            initial_transformation = np.random.rand(self.dimension, self.n_features - self.dimension)
        elif isinstance(initial_transformation, str) and initial_transformation == 'zero':
            initial_transformation = np.zeros((self.dimension, self.n_features - self.dimension))
        initial_guess = np.concatenate((initial_matrix.ravel(),
                                         initial_transformation.ravel()))
        return initial_guess
    
    def _unpack_matrix_and_transformation(self, x):
        """
        Unpack the flattened parameter vector into matrix A and transformation matrix H compatible with sklearn.

        Parameters:
        x (np.ndarray): Flattened parameter vector.

        Returns:
        tuple: Matrix A and transformation H.
        """
        A_flat = x[:self.dimension**2]
        H_flat = x[self.dimension**2:]
        A = A_flat.reshape((self.dimension, self.dimension))
        H = H_flat.reshape((self.dimension, self.n_features-self.dimension))
        H = np.hstack((np.eye(self.dimension), H))
        return A, H
    
    
    def _unpack_matrix_and_transformation_inverse(self, x):
        A_flat = x[:self.dimension**2]
        H_and_H_inv_flat = x[self.dimension**2:]
        number_of_unknowns = int(len(H_and_H_inv_flat) / 2)
        A = A_flat.reshape((self.dimension, self.dimension))
        H = H_and_H_inv_flat[:number_of_unknowns].reshape((self.dimension, self.n_features-self.dimension))
        H = np.hstack((np.eye(self.dimension), H))
        H_inv = H_and_H_inv_flat[number_of_unknowns:].reshape((self.dimension, self.n_features-self.dimension))
        H_inv = np.hstack((np.eye(self.dimension), H_inv))

        return A, H, H_inv

    def _objective_function(self, x_param,
                             x_data_nonlinear_features, # this is assumed to be a list of trajectories
                               alpha = 0.):
        """
        Create the callable objective function for the optimization. The objective function is the sum of the squared difference between A H(X)_n and H(X)_{n+1}. A regularization is added proportional to the squared Frobenius norm of the nonlinear part of the transformation coefficients.

        Parameters:
        x_param (np.ndarray): Flattened parameter vector.
        x_data_nonlinear_features (list of np.ndarray): List of nonlinear features of data.
        alpha (float): Regularization parameter.

        Returns:
        float: Value of the objective function.
        """

        A, H = self._unpack_matrix_and_transformation(x_param)
        transformed_coordinates = [np.matmul(H, n_data_nl.T) for n_data_nl in x_data_nonlinear_features]
        rhs = [np.matmul(A, tr_coords[:, :-1]) for tr_coords in transformed_coordinates]
        lhs = [tr_coords[:, 1:] for tr_coords in transformed_coordinates]
        trajectory_errors = [np.linalg.norm((rhs[i] - lhs[i]).ravel())**2 for i in range(len(rhs))]
        return np.sum(trajectory_errors) + alpha * np.linalg.norm(H[:, self.dimension:].ravel())**2 # only penalize nonlinear part 
    
    def _objective_function_lsq(self, x_param,
                             x_data_nonlinear_features, # this is assumed to be a list of trajectories
                               alpha = 0.):
        """
        Compute the objective function for least squares optimization.

        Parameters:
        x_param (np.ndarray): Flattened parameter vector.
        x_data_nonlinear_features (list of np.ndarray): List of nonlinear features of data.
        alpha (float): Regularization parameter.

        Returns:
        np.ndarray: Value of the objective function.
        """


        A, H = self._unpack_matrix_and_transformation(x_param)

        transformed_coordinates = [np.matmul(H, n_data_nl.T) for n_data_nl in x_data_nonlinear_features]
        rhs = [np.matmul(A, tr_coords[:, :-1]) for tr_coords in transformed_coordinates]
        lhs = [tr_coords[:, 1:] for tr_coords in transformed_coordinates]
        trajectory_errors = [rhs[i].ravel() - lhs[i].ravel() for i in range(len(rhs))]
        trajectory_errors.append(alpha * H[:, self.dimension:].ravel()) # regularization part
        return np.concatenate(trajectory_errors)
    
    def _objective_function_with_inverse_lsq(self, x_param,x_data,
                             x_data_nonlinear_features, # this is assumed to be a list of trajectories
                               alpha = 0.):
        A, H, H_inv = self._unpack_matrix_and_transformation_inverse(x_param)

        transformed_coordinates = [np.matmul(H, n_data_nl.T) for n_data_nl in x_data_nonlinear_features]
        nonlinear_features_for_inverse = [self.poly.fit_transform(x_single.T) for x_single in transformed_coordinates]
        inverse_transformed = [np.matmul(H_inv, n_data_nl.T) for n_data_nl in nonlinear_features_for_inverse ]
        inverse_errors = [inverse_transformed[i].ravel() - x_data[i].ravel() for i in range(len(transformed_coordinates))]

        rhs = [np.matmul(A, tr_coords[:, :-1]) for tr_coords in transformed_coordinates]
        lhs = [tr_coords[:, 1:] for tr_coords in transformed_coordinates]
        trajectory_errors = [rhs[i].ravel() - lhs[i].ravel() for i in range(len(rhs))]
        trajectory_errors.append(alpha * H[:, self.dimension:].ravel())
        trajectory_errors = np.concatenate(trajectory_errors)
        inverse_errors = np.concatenate(inverse_errors)
        
        return np.concatenate([trajectory_errors, inverse_errors])
      


    def _objective_function_torch(self, x_param, # TODO: make this work with multiple trajectories
                             x_data_nonlinear_features,
                               alpha = 0., device = None):
        A_flat = x_param[:self.dimension**2]
        H_flat = x_param[self.dimension**2:]
        A = A_flat.reshape((self.dimension, self.dimension))
        H = H_flat.reshape((self.dimension, self.n_features-self.dimension))
        H = torch.hstack((torch.eye(self.dimension, device = device), H))
        transformed_coordinates = torch.matmul(H, x_data_nonlinear_features.T)
        rhs = torch.matmul(A, transformed_coordinates[:, :-1])
        lhs = transformed_coordinates[:, 1:]
        return torch.linalg.norm((rhs - lhs).ravel())**2 + alpha * torch.linalg.norm(H[:, self.dimension:].ravel())**2
    
    def _jacobian_of_objective_function(self,
                                         x_param,
                                           x_data_nonlinear_features, 
                                           alpha = 0.):
        # here the transformation is assumed to be only the nonlinear part, so we don't add the identity matrix
        A_flat = x_param[:self.dimension**2]
        H_flat = x_param[self.dimension**2:]
        A = A_flat.reshape((self.dimension, self.dimension))
        H = H_flat.reshape((self.dimension, self.n_features-self.dimension))
        PhiX = [n_data_nl.T[self.dimension:, :-1] for n_data_nl in x_data_nonlinear_features]
        PhiY = [n_data_nl.T[self.dimension:, 1:] for n_data_nl in x_data_nonlinear_features]
        X = [n_data_nl.T[:self.dimension, :-1] for n_data_nl in x_data_nonlinear_features]
        Y = [n_data_nl.T[:self.dimension, 1:] for n_data_nl in x_data_nonlinear_features]
        lhs1 = [np.linalg.multi_dot(
            (A, X[i], X[i].T)) + np.linalg.multi_dot(
            (A, H, PhiX[i], X[i].T))  + np.linalg.multi_dot(
            (A, X[i], PhiX[i].T, H.T)) + np.linalg.multi_dot(
            (A, H, PhiX[i], PhiX[i].T, H.T)) for i in range(len(X))]
        rhs1 = [np.linalg.multi_dot(
            (Y[i], X[i].T)) + np.linalg.multi_dot(
            (H, PhiY[i], X[i].T)) + np.linalg.multi_dot(
            (Y[i], PhiX[i].T, H.T)) + np.linalg.multi_dot(
            (H, PhiY[i], PhiX[i].T, H.T)) for i in range(len(X))]

        lhs2 = [np.linalg.multi_dot(
            (A.T, A, X[i], PhiX[i].T)) + np.linalg.multi_dot(
            (A.T, A, H, PhiX[i], PhiX[i].T)) - np.linalg.multi_dot(
            (A.T, Y[i], PhiX[i].T)) - np.linalg.multi_dot(
            (A.T, H, PhiY[i], PhiX[i].T)) for i in range(len(X))]
        rhs2 = [np.linalg.multi_dot(
            (A, X[i], PhiY[i].T)) + np.linalg.multi_dot(
            (A, H, PhiX[i], PhiY[i].T)) - np.linalg.multi_dot(
            (Y[i], PhiY[i].T)) - np.linalg.multi_dot(
            (H, PhiY[i], PhiY[i].T)) for i in range(len(X))]
        lhs = [np.concatenate((lhs1[i].ravel(), lhs2[i].ravel())) for i in range(len(X))]
        rhs = [np.concatenate((rhs1[i].ravel(), rhs2[i].ravel())) for i in range(len(X))]
        lhs = sum(lhs)
        rhs = sum(rhs)
        derivative_of_regularization = x_param.ravel()
        derivative_of_regularization[:self.dimension**2] = 0.
        return -2*rhs.ravel() + 2*lhs.ravel() + 2 * alpha * derivative_of_regularization
        
    def fit(self, x, y = None, 
            alpha = 0., 
            initial_matrix = None, 
            initial_transformation = None, 
            method = 'simple',
            method_optimization = 'default', 
            verbose = False):
        """
        Fit the transformation H and the linear model, such that H(X)_{n+1} \approx A H(X)_n. The method can be either 'simple', when the squared difference between the left and right hand side of the equation is minimized, or 'with_inverse' if an additional term is added to fit the inverse transformation. The optimization is performed using the scipy minimize function.

        Parameters:
        x (list of np.ndarray): List of state matrices. They should have shape (n_dimensions, n_samples)
        y (np.ndarray): Redundant, kept for compatibility with sklearn.
        alpha (float): Regularization parameter.
        initial_matrix (np.ndarray): Initial matrix for the linear model.
        initial_transformation (np.ndarray): Initial transformation matrix.
        method (str): Sets the type of objective function to use. Can be 'simple' or 'with_inverse'.
        method_optimization (str): Optimization method to use (see scipy.optimize.minimize).
        verbose (bool): Whether to print the initial and final costs.
        """
        nonlinear_features = [self.poly.fit_transform(x_single.T) for x_single in x]
        if method == 'simple':
            initial_guess = self._prepare_initial_guess(initial_matrix, initial_transformation)
            costfunction = lambda x : self._objective_function(x, nonlinear_features, alpha = alpha)
            jac = lambda x : self._jacobian_of_objective_function(x, nonlinear_features, alpha = alpha)
            #print('Initial cost: ', costfunction(initial_guess))
            if verbose:
                print('Initial cost: ', costfunction(initial_guess))
            if method_optimization == 'default':
                method_optimization = 'BFGS'
            result = minimize(costfunction, initial_guess, method = method_optimization, jac = jac, tol = 1e-12, options = {'maxiter': 6000, 'gtol':1e-7})
            if verbose:
                print(result)
                print('Final cost: ', costfunction(result.x))

            A, H = self._unpack_matrix_and_transformation(result.x)
            self.linear_model = A
            self.transformation_coefficients = H # this only contains the nonlinear part of the transformation
            v, w = np.linalg.eig(A)
            self.transform_to_nondiagonal = w
            self.transform_to_diagonal = np.linalg.inv(w)
        if method == 'with_inverse':
            nonlinear_features = [self.poly.fit_transform(x_single.T) for x_single in x]
    
            initial_guess = self._prepare_initial_guess(initial_matrix, initial_transformation)
            initial_guess = np.concatenate((initial_guess, initial_guess[int(self.dimension**2):]))
            if method_optimization == 'default':
                method_optimization = 'trf'
            costfunction = lambda y : self._objective_function_with_inverse_lsq(y, x, 
                                                                                nonlinear_features,
                                                                                  alpha = alpha)
            
            if verbose:
                print('Initial cost: ', np.linalg.norm(costfunction(initial_guess))**2)

            result = least_squares(costfunction, initial_guess, method = method_optimization, ftol=1e-12, loss='huber')
            if verbose:
                print(result)
                print('Final cost: ', np.linalg.norm(costfunction(result.x))**2)

            A, H, H_inv = self._unpack_matrix_and_transformation_inverse(result.x)
            self.linear_model = A
            self.transformation_coefficients = H # this contains all the coefficients of the transformation
            v, w = np.linalg.eig(A)
            self.transform_to_nondiagonal = w
            self.transform_to_diagonal = np.linalg.inv(w)
            self.inverse_transformation_model = LinearRegression(fit_intercept = False)
            self.inverse_transformation_model.fit(nonlinear_features[0], x[0].T)#nonlinear_features_transformed
            self.inverse_transformation_model.coef_ = H_inv 
            self.inverse_transformation_model.n_features_in_ = self.inverse_transformation_model.coef_.shape[1] # to avoid a warning
            return
        


    def fit_torch(self, x, y = None, epochs = 1000, 
                  alpha = 0., initial_matrix = None, lr = 1e-3,
                    initial_transformation = None, device_type = 'cpu'):
        """
        Alternative implementation of DDL using pytorch. Fit the transformation H and the linear model, such that H(X)_{n+1} \approx A H(X)_n. The optimization is performed using gradient descent in pytorch. 

        Parameters:
        x (list of torch.tensors): List of state matrices. They should have shape (n_dimensions, n_samples)
        y : Redundant, kept for compatibility with sklearn.
        alpha (float): Regularization parameter.
        initial_matrix (torch.tensor): Initial matrix for the linear model.
        initial_transformation (torch.tensor): Initial transformation matrix.
        """
        device = torch.device(device_type)
        if device_type == 'mps':
            floatype = torch.float32
        else:
            floatype = torch.float64
        nonlinear_features = self.poly.fit_transform(x.T)
        nonlinear_features_tensor = torch.tensor(nonlinear_features, device = device, dtype = floatype)    

        dmd_model = dmd([x])
        A0 = dmd_model.coef_
        H0 = np.zeros((self.dimension, self.n_features - self.dimension))
        initial_guess = np.concatenate((A0.ravel(),
                                         H0.ravel()))
        initial_guess = torch.tensor(initial_guess, requires_grad=True, device = device, dtype = floatype)
        costfunction = lambda x : self._objective_function_torch(x, nonlinear_features_tensor, alpha = alpha, device = device)
        optimizer = torch.optim.Adam([initial_guess], lr=lr)
        # Optimization loop
        print('Initial cost: ', costfunction(initial_guess).cpu().detach().numpy())
        for _ in range(epochs):
            optimizer.zero_grad()
            cost = costfunction(initial_guess)
            cost.backward()
            optimizer.step()
        A, H = self._unpack_matrix_and_transformation(initial_guess.detach().numpy())

        print('Final cost: ', costfunction(initial_guess).cpu().detach().numpy())
        self.linear_model = A
        self.transformation_coefficients = H  # This only contains the nonlinear part of the transformation
        return
    
    def fit_inverse(self, x, y = None):
        
        if self.linear_model is None:
            raise ValueError('The model has not been fitted yet. Call fit() first.')
        X = np.concatenate(x, axis = 1)
        transformed_coordinates = self.transform(X)
        self.inverse_transformation_model = LinearRegression(fit_intercept=False)
        nonlinear_features_transformed = self.poly.fit_transform(transformed_coordinates.T)[:,self.dimension:] # fit only the nonlinear part
        self.inverse_transformation_model.fit(
            nonlinear_features_transformed, X.T - transformed_coordinates.T)
        self.inverse_transformation_model.coef_ = np.hstack((np.eye(self.dimension), self.inverse_transformation_model.coef_))
        self.inverse_transformation_model.n_features_in_ = self.inverse_transformation_model.coef_.shape[1] # to avoid a warning
        return 
    
    def transformation_to_diagonal(self, x):
        if self.transform_to_diagonal is None:
            raise ValueError('The model has not been fitted yet. Call fit() first.')
        return np.matmul(self.transform_to_diagonal, x)
    
    def transformation_to_nondiagonal(self, x):
        if self.transform_to_nondiagonal is None:
            raise ValueError('The model has not been fitted yet. Call fit() first.')
        return np.matmul(self.transform_to_nondiagonal, x)
    

    def transform(self, x):
        if self.transformation_coefficients is None:
            raise ValueError('The model has not been fitted yet. Call fit() first.')
        nonlinear_features = self.poly.fit_transform(x.T)
        return np.matmul(self.transformation_coefficients, nonlinear_features.T) # to return the same shape as x (self.dimension, n_samples)
    
    def inverse_transform(self, x):
        if self.inverse_transformation_model is None:
            raise ValueError('The inverse transformation has not been fitted yet. Call fit_inverse() first.')
        nonlinear_features = self.poly.fit_transform(x.T)#[:,self.dimension:]
        return self.inverse_transformation_model.predict(nonlinear_features).T


    def predict(self, x, iterations, with_transform = True):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if self.linear_model is None:
            raise ValueError('The model has not been fitted yet. Call fit() first.')
        if with_transform:
            x = self.transform(x)
        predictions = [x]
        for _ in range(iterations):
            predictions.append(np.matmul(self.linear_model, predictions[-1]))
        return np.squeeze(np.array(predictions)).T
    
    def compute_symbolic_transforms(self):
        if self.transformation_coefficients is None:
            raise ValueError('The model has not been fitted yet. Call fit() first.')
        polynomials = self.poly
        coefficients = self.transformation_coefficients
        transform_to_linearized = {}
        transform_to_linearized['transform'], transform_to_linearized['jacobian'], transform_to_linearized['variables'] = differentiate_model_symbolic(polynomials, coefficients)
        transform_to_linearized['transform_numpy'] = sym.lambdify([transform_to_linearized['variables']],
                                                                   transform_to_linearized['transform'], 
                                                                   'numpy')
        transform_to_linearized['transform_jacobian_numpy'] = sym.lambdify([transform_to_linearized['variables']],
                                                                   transform_to_linearized['jacobian'], 
                                                                   'numpy')

        inverse_coefficients = self.inverse_transformation_model.coef_
        transform_to_original = {}
        transform_to_original['transform'], transform_to_original['jacobian'], transform_to_original['variables'] = differentiate_model_symbolic (polynomials, inverse_coefficients)
        transform_to_original['transform_numpy'] = sym.lambdify([transform_to_original['variables']],
                                                                   transform_to_original['transform'], 
                                                                   'numpy')
        transform_to_original['transform_jacobian_numpy'] = sym.lambdify([transform_to_original['variables']],
                                                                   transform_to_original['jacobian'], 
                                                                   'numpy')
        self.symbolic_transformation_to_linearized = transform_to_linearized
        self.symbolic_transformation_to_original = transform_to_original
        return 

    def score(self, x):
        if self.linear_model is None:
            raise ValueError('The model has not been fitted yet. Call fit() first.')
        nonlinear_features = [self.poly.fit_transform(x_single.T) for x_single in x] # objective function expects this as (n_samples, n_features)
        x_param = np.concatenate((self.linear_model.ravel(), self.transformation_coefficients[:, self.dimension:].ravel()))
        return self._objective_function(x_param, nonlinear_features, alpha = 0.)
    
