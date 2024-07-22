import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize, least_squares
from sklearn.linear_model import LinearRegression, Ridge
import torch 
from torch.autograd.functional import hessian, jacobian
import sympy as sym
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)


def DMD(x):
    """DMD model with no regularization"""
    XX = [x_element[:, :-1].T for x_element in x]
    YY = [x_element[:, 1:].T for x_element in x]
    XX = np.concatenate(XX)
    YY = np.concatenate(YY)
    DMDModel = LinearRegression(fit_intercept=False)
    DMDModel.fit(XX, YY)
    return DMDModel

def EDMD(x, degree = 3):
    """EDMD model with no regularization and polynomial features"""

    nonlinear_features = [PolynomialFeatures(degree = degree, include_bias=False).fit_transform(x_element.T) for x_element in x]
    XX_EDMD = [nonlinear_feature_element[:-1, :] for nonlinear_feature_element in nonlinear_features]
    YY_EDMD = [nonlinear_feature_element[1:, :] for nonlinear_feature_element in nonlinear_features]
    XX_EDMD = np.concatenate(XX_EDMD)
    YY_EDMD = np.concatenate(YY_EDMD)
    EDMDModel = LinearRegression(fit_intercept=False)
    EDMDModel.fit(XX_EDMD, YY_EDMD)
    return EDMDModel


def differentiate_model(polynomials, coefficients, x):
    """Differentiate a model with polynomial features"""
    # x is assumed to be of shape (n_features, n_samples)
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

class DataDrivenLinearization:
    def __init__(self, dimension, degree = 3, approximate = False):
        self.degree = degree
        self.is_approximate = approximate
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
        if initial_matrix is None:
            initial_matrix = np.random.rand(self.dimension, self.dimension)
        if initial_transformation is None:
            initial_transformation = np.random.rand(self.dimension, self.n_features - self.dimension)
        if initial_transformation == 'zero':
            initial_transformation = np.zeros((self.dimension, self.n_features - self.dimension))
        initial_guess = np.concatenate((initial_matrix.ravel(),
                                         initial_transformation.ravel()))
        return initial_guess
    

    def _prepare_initial_guess_fixed_matrix(self, 
                                            initial_transformation = None):
        if initial_transformation is None:
            initial_transformation = np.random.rand(self.dimension, self.n_features - self.dimension)
        if initial_transformation == 'zero':
            initial_transformation = np.zeros((self.dimension, self.n_features - self.dimension))
        initial_guess = initial_transformation.ravel()
        return initial_guess

    
    def _unpack_matrix_and_transformation(self, x):
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
    
    def _unpack_matrix_and_transformation_fixed_matrix(self, x):
        # this is used when we want to keep the linear model fixed
        H = x.reshape((self.dimension, self.n_features-self.dimension))
        H = np.hstack((np.eye(self.dimension), H))
        return H
    


    def _objective_function(self, x_param,
                             x_data_nonlinear_features, # this is assumed to be a list of trajectories
                               alpha = 0.):
        A, H = self._unpack_matrix_and_transformation(x_param)

        transformed_coordinates = [np.matmul(H, n_data_nl.T) for n_data_nl in x_data_nonlinear_features]
        rhs = [np.matmul(A, tr_coords[:, :-1]) for tr_coords in transformed_coordinates]
        lhs = [tr_coords[:, 1:] for tr_coords in transformed_coordinates]
        trajectory_errors = [np.linalg.norm((rhs[i] - lhs[i]).ravel())**2 for i in range(len(rhs))]
        return np.sum(trajectory_errors) + alpha * np.linalg.norm(H[:, self.dimension:].ravel())**2 # only penalize nonlinear part 
    def _objective_function_lsq(self, x_param,
                             x_data_nonlinear_features, # this is assumed to be a list of trajectories
                               alpha = 0.):
        A, H = self._unpack_matrix_and_transformation(x_param)

        transformed_coordinates = [np.matmul(H, n_data_nl.T) for n_data_nl in x_data_nonlinear_features]
        rhs = [np.matmul(A, tr_coords[:, :-1]) for tr_coords in transformed_coordinates]
        lhs = [tr_coords[:, 1:] for tr_coords in transformed_coordinates]
        trajectory_errors = [rhs[i].ravel() - lhs[i].ravel() for i in range(len(rhs))]
        trajectory_errors.append(alpha * H[:, self.dimension:].ravel())
        return np.concatenate(trajectory_errors)#np.sum(trajectory_errors) + alpha * np.linalg.norm(H[:, self.dimension:].ravel())**2 # only penalize nonlinear part 
    
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
        
        return np.concatenate([trajectory_errors, inverse_errors])#np.sum(trajectory_errors) + alpha * np.linalg.norm(H[:, self.dimension:].ravel())**2 # only penalize nonlinear part 
        
    def _objective_function_fixed_matrix_lsq(self, x_param,
                             x_data_nonlinear_features, # this is assumed to be a list of trajectories
                               alpha = 0.):
        H = self._unpack_matrix_and_transformation_fixed_matrix(x_param)
        A = self.linear_model # has to be specified before calling this function
        transformed_coordinates = [np.matmul(H, n_data_nl.T) for n_data_nl in x_data_nonlinear_features]
        rhs = [np.matmul(A, tr_coords[:, :-1]) for tr_coords in transformed_coordinates]
        lhs = [tr_coords[:, 1:] for tr_coords in transformed_coordinates]
        trajectory_errors = [rhs[i].ravel() - lhs[i].ravel() for i in range(len(rhs))]
        trajectory_errors.append(alpha * H[:, self.dimension:].ravel())
        return np.concatenate(trajectory_errors)#np.sum(trajectory_errors) + alpha * np.linalg.norm(H[:, self.dimension:].ravel())**2 # only penalize nonlinear part 


    def _objective_function_fixed_matrix(self, x_param,
                             x_data_nonlinear_features, # this is assumed to be a list of trajectories
                               alpha = 0.):
        H = self._unpack_matrix_and_transformation_fixed_matrix(x_param)
        transformed_coordinates = [np.matmul(H, n_data_nl.T) for n_data_nl in x_data_nonlinear_features]
        A = self.linear_model # has to be specified before calling this function
        rhs = [np.matmul(A, tr_coords[:, :-1]) for tr_coords in transformed_coordinates]
        lhs = [tr_coords[:, 1:] for tr_coords in transformed_coordinates]
        trajectory_errors = [np.linalg.norm((rhs[i] - lhs[i]).ravel())**2 for i in range(len(rhs))]
        return np.sum(trajectory_errors) + alpha * np.linalg.norm(H[:,self.dimension:].ravel())**2 # only penalize nonlinear part 
    
    def _objective_function_advected_discrete(self, x_param,
                             x_data_nonlinear_features, # this is assumed to be a list of trajectories
                               alpha = 0., time_eval = None):
        A, H = self._unpack_matrix_and_transformation(x_param)
        transformed_coordinates = [np.matmul(H, n_data_nl.T) for n_data_nl in x_data_nonlinear_features]
        #A = self.linear_model # has to be specified before calling this function
        # let's assume for this function that A is the continuous time model
        advected_coords = []
        for transformed_traj in transformed_coordinates:
            times_max = transformed_traj.shape[1]
            A_matrix_powers = np.array([np.linalg.matrix_power(A,i) for i in range(times_max)])
            trajectory = A_matrix_powers@transformed_traj[:,0]
            advected_coords.append(np.array(trajectory).T)
        trajectory_errors = [transformed_coordinates[i].ravel() - advected_coords[i].ravel() for i in range(len(transformed_coordinates))]
        return np.concatenate(trajectory_errors)#np.sum(trajectory_errors) + alpha * np.linalg.norm(H[:,self.dimension:].ravel())**2 # only penalize nonlinear part 



    def _objective_function_advected_discrete_with_inverse(self, x_param, x_data,
                             x_data_nonlinear_features, # this is assumed to be a list of trajectories
                               alpha = 0., time_eval = None):
        A, H, H_inv = self._unpack_matrix_and_transformation_inverse(x_param)
        transformed_coordinates = [np.matmul(H, n_data_nl.T) for n_data_nl in x_data_nonlinear_features]
        advected_coords = []
        for transformed_traj in transformed_coordinates:
            times_max = transformed_traj.shape[1]
            A_matrix_powers = np.array([np.linalg.matrix_power(A,i) for i in range(times_max)])
            trajectory = A_matrix_powers@transformed_traj[:,0]
            advected_coords.append(np.array(trajectory).T)
        nonlinear_features_for_inverse = [self.poly.fit_transform(x_single.T) for x_single in advected_coords]
        inverse_transformed = [np.matmul(H_inv, n_data_nl.T) for n_data_nl in nonlinear_features_for_inverse ]
        #print(inverse_transformed[0].shape, x_data[0].shape)
        inverse_errors = [inverse_transformed[i].ravel() - x_data[i].ravel() for i in range(len(transformed_coordinates))]
        trajectory_errors = [transformed_coordinates[i].ravel() - advected_coords[i].ravel() for i in range(len(transformed_coordinates))]
        trajectory_errors.append(alpha * H[:, self.dimension:].ravel())
        inverse_errors.append(alpha * H_inv[:, self.dimension:].ravel())
        inverse_errors = np.concatenate(inverse_errors)
        trajectory_errors = np.concatenate(trajectory_errors)

        return np.concatenate((inverse_errors, trajectory_errors))
    
    def _objective_function_fixed_matrix_advected_discrete(self, x_param,
                             x_data_nonlinear_features, # this is assumed to be a list of trajectories
                               alpha = 0., time_eval = None):
        H = self._unpack_matrix_and_transformation_fixed_matrix(x_param)
        transformed_coordinates = [np.matmul(H, n_data_nl.T) for n_data_nl in x_data_nonlinear_features]
        A = self.linear_model # has to be specified before calling this function
        # let's assume for this function that A is the continuous time model
        advected_coords = []
        for transformed_traj in transformed_coordinates:
            times_max = transformed_traj.shape[1]
            A_matrix_powers = np.array([np.linalg.matrix_power(A,i) for i in range(times_max)])
            trajectory = A_matrix_powers@transformed_traj[:,0]
            advected_coords.append(np.array(trajectory).T)
        #print(transformed_coordinates[0][:,0])
        #err = np.linalg.norm(transformed_coordinates[0] - advected_coords[0], axis = 0)
        #plt.semilogy(err)
        #plt.plot(transformed_coordinates[0][0,:])
        ##plt.plot(advected_coords[0][0,:], '.')
        #plt.xlim(350, 400)
        #plt.plot(transformed_coordinates[0][0,:], transformed_coordinates[0][0,:], '-')
        trajectory_errors = [transformed_coordinates[i].ravel() - advected_coords[i].ravel() for i in range(len(transformed_coordinates))]
        return np.concatenate(trajectory_errors)#np.sum(trajectory_errors) + alpha * np.linalg.norm(H[:,self.dimension:].ravel())**2 # only penalize nonlinear part 
    
    
    


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
            method = 'BFGS'):
        # Fit the transformation H and the linear model, such that H(X)_{n+1} \approx A H(X)_n
        # x is assumed to be of shape (self.dimension, n_samples)
        # y is redundant, but kept for compatibility with sklearn
        # alpha is a regularization parameter
        # if the approximate flag is set to True, then we solve the linearized problem around the DMD solution
        nonlinear_features = [self.poly.fit_transform(x_single.T) for x_single in x]
        if self.is_approximate:
            dmd = DMD(x)
            A0 = torch.tensor(dmd.coef_, requires_grad = True)
            nonlinear_features_tensor = torch.tensor(nonlinear_features)    
            H0 = torch.zeros((self.dimension, self.n_features - self.dimension),  requires_grad = True)
            initial_guess = torch.cat((A0.ravel(), H0.ravel()))
            toopt = lambda x : self._objective_function_torch(x, nonlinear_features_tensor, alpha = alpha)
            HH = hessian(toopt, initial_guess)
            JJ = jacobian(toopt, initial_guess)
            sol = torch.linalg.solve(HH, -JJ)
            solnp = sol.detach().numpy()
            A, H = self._unpack_matrix_and_transformation(solnp)
            self.linear_model = A + A0.detach().numpy()
            self.transformation_coefficients = H
            return 
        else:
            initial_guess = self._prepare_initial_guess(initial_matrix, initial_transformation)
            costfunction = lambda x : self._objective_function(x, nonlinear_features, alpha = alpha)
            jac = lambda x : self._jacobian_of_objective_function(x, nonlinear_features, alpha = alpha)
            #print('Initial cost: ', costfunction(initial_guess))

            result = minimize(costfunction, initial_guess, method = method, jac = jac, tol = 1e-12, options = {'maxiter': 6000, 'gtol':1e-7})
            #print(result)
            #print('Final cost: ', costfunction(result.x))

            A, H = self._unpack_matrix_and_transformation(result.x)
            self.linear_model = A
            self.transformation_coefficients = H # this only contains the nonlinear part of the transformation
            v, w = np.linalg.eig(A)
            self.transform_to_nondiagonal = w
            self.transform_to_diagonal = np.linalg.inv(w)
            return
        

    def fit_lsq(self, x, y = None, 
            alpha = 0., 
            initial_matrix = None, 
            initial_transformation = None, 
            method = 'lm', verbose = True):
        # Fit the transformation H and the linear model, such that H(X)_{n+1} \approx A H(X)_n
        # x is assumed to be of shape (self.dimension, n_samples)
        # y is redundant, but kept for compatibility with sklearn
        # alpha is a regularization parameter
        # if the approximate flag is set to True, then we solve the linearized problem around the DMD solution
        nonlinear_features = [self.poly.fit_transform(x_single.T) for x_single in x]
    
        initial_guess = self._prepare_initial_guess(initial_matrix, initial_transformation)
        costfunction = lambda x : self._objective_function_lsq(x, nonlinear_features, alpha = alpha)
        jac = lambda x : self._jacobian_of_objective_function(x, nonlinear_features, alpha = alpha)
        if verbose:
            print('Initial cost: ', np.linalg.norm(costfunction(initial_guess))**2)

        result = least_squares(costfunction, initial_guess, method = method, ftol=1e-15, loss='linear')
        if verbose:
            print(result)
            print('Final cost: ', np.linalg.norm(costfunction(result.x))**2)

        A, H = self._unpack_matrix_and_transformation(result.x)
        self.linear_model = A
        self.transformation_coefficients = H # this only contains the nonlinear part of the transformation
        v, w = np.linalg.eig(A)
        self.transform_to_nondiagonal = w
        self.transform_to_diagonal = np.linalg.inv(w)
        return np.linalg.norm(costfunction(result.x))**2 # return the final cost


    def fit_lsq_inv(self, x, y = None, 
            alpha = 0., 
            initial_matrix = None, 
            initial_transformation = None, 
            method = 'lm', verbose = True):
        # Fit the transformation H and the linear model, such that H(X)_{n+1} \approx A H(X)_n
        # x is assumed to be of shape (self.dimension, n_samples)
        # y is redundant, but kept for compatibility with sklearn
        # alpha is a regularization parameter
        # if the approximate flag is set to True, then we solve the linearized problem around the DMD solution
        nonlinear_features = [self.poly.fit_transform(x_single.T) for x_single in x]
    
        initial_guess = self._prepare_initial_guess(initial_matrix, initial_transformation)
        initial_guess = np.concatenate((initial_guess, initial_guess[int(self.dimension**2):]))
        costfunction = lambda y : self._objective_function_with_inverse_lsq(y, x, nonlinear_features, alpha = alpha)
        jac = lambda x : self._jacobian_of_objective_function(x, nonlinear_features, alpha = alpha)
        if verbose:
            print('Initial cost: ', np.linalg.norm(costfunction(initial_guess))**2)

        result = least_squares(costfunction, initial_guess, method = method, ftol=1e-12,loss='huber')
        if verbose:
            print(result)
            print('Final cost: ', np.linalg.norm(costfunction(result.x))**2)

        A, H, H_inv = self._unpack_matrix_and_transformation_inverse(result.x)
        self.linear_model = A
        self.transformation_coefficients = H # this only contains the nonlinear part of the transformation
        v, w = np.linalg.eig(A)
        self.transform_to_nondiagonal = w
        self.transform_to_diagonal = np.linalg.inv(w)
        self.inverse_transformation_model = LinearRegression(fit_intercept=False)
        self.inverse_transformation_model.fit(nonlinear_features[0], x[0].T)#nonlinear_features_transformed
        self.inverse_transformation_model.coef_ = H_inv#np.hstack((np.eye(self.dimension), H_inv))
        self.inverse_transformation_model.n_features_in_ = self.inverse_transformation_model.coef_.shape[1] # to avoid a warning
     
        return np.linalg.norm(costfunction(result.x))**2 # return the final cost
    
    


    def fit_fixed_matrix_lsq(self, x, initial_matrix, y = None, 
        alpha = 0., 
        initial_transformation = None, 
        method = 'lm', verbose = True):
    # Fit the transformation H and the linear model, such that H(X)_{n+1} \approx A H(X)_n
    # x is assumed to be of shape (self.dimension, n_samples)
    # y is redundant, but kept for compatibility with sklearn
    # alpha is a regularization parameter
    # if the approximate flag is set to True, then we solve the linearized problem around the DMD solution
        self.linear_model = initial_matrix

        nonlinear_features = [self.poly.fit_transform(x_single.T) for x_single in x]
        initial_guess = self._prepare_initial_guess_fixed_matrix(initial_transformation)
        costfunction = lambda x : self._objective_function_fixed_matrix_lsq(x, nonlinear_features, alpha = alpha)
        #jac = lambda x : self._jacobian_of_objective_function(x, nonlinear_features, alpha = alpha)
        if verbose: 
            print('Initial cost: ', np.linalg.norm(costfunction(initial_guess))**2)

        result = least_squares(costfunction, initial_guess, method = method)
        if verbose:
            print(result)
            print('Final cost: ', np.linalg.norm(costfunction(result.x))**2)

        H = self._unpack_matrix_and_transformation_fixed_matrix(result.x)
        self.transformation_coefficients = H # this only contains the nonlinear part of the transformation
        return np.linalg.norm(costfunction(result.x))**2
    
    def fit_fixed_matrix_fullpred_discrete(self, x, initial_matrix, y = None, 
        alpha = 0., 
        initial_transformation = None, 
        method = 'lm', time_eval = None, verbose = False):
    # Fit the transformation H and the linear model, such that H(X)_{n+1} \approx A H(X)_n
    # x is assumed to be of shape (self.dimension, n_samples)
    # y is redundant, but kept for compatibility with sklearn
    # alpha is a regularization parameter
    # if the approximate flag is set to True, then we solve the linearized problem around the DMD solution
        self.linear_model = initial_matrix

        nonlinear_features = [self.poly.fit_transform(x_single.T) for x_single in x]
        initial_guess = self._prepare_initial_guess_fixed_matrix(initial_transformation)
        costfunction = lambda x : self._objective_function_fixed_matrix_advected_discrete(x, nonlinear_features, alpha = alpha, time_eval=time_eval)
        #jac = lambda x : self._jacobian_of_objective_function(x, nonlinear_features, alpha = alpha)
        if verbose:
            print('Initial cost: ', np.linalg.norm(costfunction(initial_guess))**2)
        result = least_squares(costfunction, initial_guess, method = method, ftol=1e-15)

#        result = minimize(costfunction, initial_guess, method = method, tol = 1e-10, options = {'maxiter': 20000, 'maxfev': 20000})
        if verbose:
            print(result)
            print('Final cost: ', np.linalg.norm(costfunction(result.x))**2)

        H = self._unpack_matrix_and_transformation_fixed_matrix(result.x)
        self.transformation_coefficients = H # this only contains the nonlinear part of the transformation
        return np.linalg.norm(costfunction(result.x))**2

    def fit_fullpred_discrete(self, x, initial_matrix=None, y = None, 
        alpha = 0., 
        initial_transformation = None, 
        method = 'lm', time_eval = None, verbose = False):
    # Fit the transformation H and the linear model, such that H(X)_{n+1} \approx A H(X)_n
    # x is assumed to be of shape (self.dimension, n_samples)
    # y is redundant, but kept for compatibility with sklearn
    # alpha is a regularization parameter
    # if the approximate flag is set to True, then we solve the linearized problem around the DMD solution

        nonlinear_features = [self.poly.fit_transform(x_single.T) for x_single in x]
        initial_guess = self._prepare_initial_guess(initial_matrix, initial_transformation)
        costfunction = lambda x : self._objective_function_advected_discrete(x, nonlinear_features, alpha = alpha, time_eval=time_eval)
        #jac = lambda x : self._jacobian_of_objective_function(x, nonlinear_features, alpha = alpha)
        if verbose:
            print('Initial cost: ', np.linalg.norm(costfunction(initial_guess))**2)
        result = least_squares(costfunction, initial_guess, method = method, ftol=1e-15)

#        result = minimize(costfunction, initial_guess, method = method, tol = 1e-10, options = {'maxiter': 20000, 'maxfev': 20000})
        if verbose:
            print(result)
            print('Final cost: ', np.linalg.norm(costfunction(result.x))**2)

        A, H = self._unpack_matrix_and_transformation(result.x)
        self.transformation_coefficients = H # this only contains the nonlinear part of the transformation
        self.linear_model = A

        return np.linalg.norm(costfunction(result.x))**2

    def fit_fullpred_discrete_inv(self, x, initial_matrix=None, y = None, 
        alpha = 0., 
        initial_transformation = None, 
        method = 'lm', time_eval = None, verbose = False):
    # Fit the transformation H and the linear model, such that H(X)_{n+1} \approx A H(X)_n
    # x is assumed to be of shape (self.dimension, n_samples)
    # y is redundant, but kept for compatibility with sklearn
    # alpha is a regularization parameter
    # if the approximate flag is set to True, then we solve the linearized problem around the DMD solution

        nonlinear_features = [self.poly.fit_transform(x_single.T) for x_single in x]
        initial_guess = self._prepare_initial_guess(initial_matrix, initial_transformation)
        initial_guess = np.concatenate((initial_guess, initial_guess[int(self.dimension**2):]))
        costfunction = lambda y : self._objective_function_advected_discrete_with_inverse(y, x, nonlinear_features, alpha = alpha)
        if verbose:
            print('Initial cost: ', np.linalg.norm(costfunction(initial_guess))**2)
        result = least_squares(costfunction, initial_guess, method = method, ftol=1e-15)

#        result = minimize(costfunction, initial_guess, method = method, tol = 1e-10, options = {'maxiter': 20000, 'maxfev': 20000})
        if verbose:
            print(result)
            print('Final cost: ', np.linalg.norm(costfunction(result.x))**2)
        A, H, H_inv = self._unpack_matrix_and_transformation_inverse(result.x)
        self.linear_model = A
        self.transformation_coefficients = H # this only contains the nonlinear part of the transformation
        v, w = np.linalg.eig(A)
        self.transform_to_nondiagonal = w
        self.transform_to_diagonal = np.linalg.inv(w)
        self.inverse_transformation_model = LinearRegression(fit_intercept=False)
        self.inverse_transformation_model.fit(nonlinear_features[0], x[0].T)#nonlinear_features_transformed
        self.inverse_transformation_model.coef_ = H_inv#np.hstack((np.eye(self.dimension), H_inv))
        self.inverse_transformation_model.n_features_in_ = self.inverse_transformation_model.coef_.shape[1] # to avoid a warning
        return np.linalg.norm(costfunction(result.x))**2 # return the final cost
    

    def fit_torch(self, x, y = None, epochs = 1000, 
                  alpha = 0., initial_matrix = None, lr = 1e-3,
                    initial_transformation = None, device_type = 'cpu'):
        device = torch.device(device_type)
        if device_type == 'mps':
            floatype = torch.float32
        else:
            floatype = torch.float64
        nonlinear_features = self.poly.fit_transform(x.T)
        nonlinear_features_tensor = torch.tensor(nonlinear_features, device = device, dtype = floatype)    

#        initial_guess = self._prepare_initial_guess(initial_matrix, initial_transformation)
        dmd = DMD(x)
        A0 = dmd.coef_#torch.tensor(dmd.coef_, requires_grad = True)
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
        # here x is the trajectory data, y is redundant.

        if self.linear_model is None:
            raise ValueError('The model has not been fitted yet. Call fit() first.')
        X = np.concatenate(x, axis = 1)
        transformed_coordinates = self.transform(X)
        self.inverse_transformation_model = LinearRegression(fit_intercept=False)
        nonlinear_features_transformed = self.poly.fit_transform(transformed_coordinates)[:,self.dimension:] # fit only the nonlinear part
        #print(x.shape, transformed_coordinates.shape, nonlinear_features_transformed.shape)
        self.inverse_transformation_model.fit(
            nonlinear_features_transformed, X.T - transformed_coordinates)
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
        return np.matmul(self.transformation_coefficients, nonlinear_features.T).T # to return the same shape as x (self.dimension, n_samples)
    
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
            x = self.transform(x).T
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
    


import logging

logger = logging.getLogger("coordinates_embedding")

def coordinates_embedding(
        t: list, 
        x: list, 
        imdim: int=None, 
        over_embedding: int=0,
        force_embedding: bool=False,
        time_stepping: int=1,
        shift_steps: int=1
    ):
    """
    Adopted from SSMLearnPy at https://github.com/haller-group/SSMLearnPy
    Returns the n-dim. time series x into a time series of properly embedded
    coordinate system y of dimension p. Optional inputs to be specified as
    'field_name','field value'
        
    Parameters:
    t : list of time vectors
    x : list of observed trajectories 
    imdim - dimension of the invariant manifold to learn
        
    over_embedding (optional): augment the minimal embedding dimension with a number of
                     time delayed measurements, default 0
    force_embedding (optional): force the embedding in the states of x, default false
    time_stepping   (optional): time stepping in the time series, default 1
    shift_steps     (optional): number of timesteps passed between components (but 
                     subsequent measurements are kept intact), default 1

    Returns:
    t_y : list of time vectors

    y : cell array of dimension (N_traj,2) where the first column contains
        time instances (1 x mi each) and the second column the trajectories
        (p x mi each)
    opts_embdedding : options containing the embedding information

    """
    if not imdim:
        raise RuntimeError("imdim not specified for coordinates embedding")
    n_observables = x[0].shape[0] 
    n_n = int(np.ceil( (2*imdim + 1)/n_observables) + over_embedding)

    # Construct embedding coordinate system
    if n_n > 1 and force_embedding != 1:
        p = n_n * n_observables
        # Augment embdedding dimension with time delays
        if n_observables == 1:
            logger.info((
                f'The {str(p)} embedding coordinates consist of the ' +
                f'measured state and its {str(n_n-1)} time-delayed measurements.'
            ))
        else:
            logger.info((
                f'The {str(p)} embedding coordinates consist of the {str(n_observables)} ' +
                f'measured states and their {str(n_n-1)} time-delayed measurements.'
            ))
        t_y = []
        y = []
        for i_traj in range(len(x)):
            t_i = t[i_traj]
            x_i = x[i_traj]

            subsample = np.arange(start=0, stop=len(t_i), step=time_stepping)

            y_i = x_i[:, subsample]
            y_base = x_i[:, subsample]

            for i_rep in range(1, n_n):
                y_i = np.concatenate(
                    (
                        y_i,
                        np.roll(y_base, -i_rep) 
                    )   
                )
            
            y.append(
                y_i[:, :-n_n+1]
            )
            t_y.append(
                t_i[
                    subsample[:-n_n+1]
                ]
            )

    else:
        p = n_observables

        if time_stepping > 1:
            logger.info('The embedding coordinates consist of the measured states.')
            t_y = []
            y = []
            for i_traj in range(len(x)):
                t_i = t[i_traj]
                x_i = x[i_traj]
                subsample = np.arange(start=0, stop=len(t_i), step=time_stepping)
                t_y.append(t_i[subsample])
                y.append(x_i[:, subsample])

        else:
            t_y = t
            y = x

    opts_embdedding = {
        'imdim' : imdim,
        'over_embedding': over_embedding,
        'force_embedding': force_embedding,
        'time_stepping' : time_stepping,
        'shift_steps' : shift_steps,
        'embedding_space_dim': p
    }
    

    return t_y, y, opts_embdedding