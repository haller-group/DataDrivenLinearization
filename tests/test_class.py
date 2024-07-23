from data_driven_linearization.linearization import DataDrivenLinearization
from data_driven_linearization.differentiation_utils import differentiate_model
import numpy as np
import torch
from torch.autograd.functional import hessian, jacobian
import matplotlib.pyplot as plt
from scipy.linalg import expm
import sympy as sy
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
def test_ddl_class_ic_guess():
    d = DataDrivenLinearization(2, degree = 3)
    assert d.degree == 3
    assert d.dimension == 2
    x = np.random.rand(2,100)
    ic_guess = d._prepare_initial_guess()
    assert ic_guess.shape[0] == 2 * 2 + (9-2) * 2

def test_ddl_class_unpack_matrix_and_transformation():
    d = DataDrivenLinearization(4, degree = 3)
    assert d.n_features == 34
    x = np.random.rand(4,100)
    init_A = np.random.rand(4,4)
    init_H = np.random.rand(4,d.n_features-4)
    init_guess = d._prepare_initial_guess(init_A, init_H)
    unpacked_A, unpacked_H = d._unpack_matrix_and_transformation(init_guess)
    assert np.allclose(init_A, unpacked_A)
    assert np.allclose(np.hstack((np.eye(4), init_H)), unpacked_H)

def test_ddl_class_objective_function_and_derivative():
    np.random.seed(10)
    d = DataDrivenLinearization(4, degree = 3)
    init_A = np.random.rand(4,4)
    init_H = np.random.rand(4,d.n_features-4)
    init_guess = d._prepare_initial_guess(init_A, init_H)
    x = np.random.rand(4,100)
    nonlinear_features = d.poly.fit_transform(x.T)
    nonlinear_features_tensor = torch.tensor(nonlinear_features)
    init_guess_tensor = torch.tensor(init_guess, requires_grad=True)
    alpha = 0.1
    nA = 4
    def torch_fun(x):
        A = x[:nA**2].reshape((nA, nA))
        H = x[nA**2:].reshape((nA, d.n_features-nA))
        H = torch.hstack((torch.eye(nA), H))
        transformed_coordinates = torch.matmul(H, nonlinear_features_tensor.T)
        rhs = torch.matmul(A, transformed_coordinates[:, :-1])
        lhs = transformed_coordinates[:, 1:]
        return torch.linalg.norm((rhs - lhs).ravel())**2 + alpha * torch.linalg.norm(H.ravel())**2
    JJ = jacobian(torch_fun, init_guess_tensor)
    JJ = JJ.detach().numpy()
    J = d._jacobian_of_objective_function(init_guess, [nonlinear_features], alpha = alpha)
    assert np.allclose(J, JJ, atol=1e-4)

def test_ddl_class_fit():
    np.random.seed(10)
    d = DataDrivenLinearization(2, degree = 3)
    True_A = np.array([[-0.001, -0.02], [0.02, -0.001]])
    ic = np.array([1, 1])

    traj = [ic]
    for _ in range(1000):
        traj.append(expm(True_A) @ traj[-1])
    trajectory = [np.array(traj).T]
    d.fit(trajectory)
    eigs_true = np.linalg.eig(expm(True_A))[0]
    eigs_predicted = np.linalg.eig(d.linear_model)[0]
    assert np.allclose(np.log(eigs_true), np.log(eigs_predicted), atol=1e-5)


def test_ddl_class_fit_inv():
    np.random.seed(10)
    d = DataDrivenLinearization(2, degree = 3)
    True_A = np.array([[-0.001, -0.02], [0.02, -0.001]])
    ic = np.array([1, 1])

    traj = [ic]
    for _ in range(1000):
        traj.append(expm(True_A) @ traj[-1])
    trajectory = [np.array(traj).T]
    d.fit(trajectory, method = "with_inverse")
    eigs_true = np.linalg.eig(expm(True_A))[0]
    eigs_predicted = np.linalg.eig(d.linear_model)[0]
    assert np.allclose(np.log(eigs_true), np.log(eigs_predicted), atol=1e-5)


def test_ddl_class_transform_predict():
    np.random.seed(10)
    d = DataDrivenLinearization(2, degree = 2)
    True_A = np.array([[-0.001, -0.02], [0.02, -0.001]])
    ic = np.array([1, 1])

    traj = [ic]
    for _ in range(1000):
        traj.append(expm(True_A) @ traj[-1])
    trajectory = [np.array(traj).T]
    d.fit(trajectory, initial_transformation='zero')
    transformed = d.transform(trajectory[0])
    predicted = d.predict(trajectory[0][:,0], 1000)

    eigs_true = np.linalg.eig(expm(True_A))[0]
    eigs_predicted = np.linalg.eig(d.linear_model)[0]
    assert np.allclose(np.log(eigs_true), np.log(eigs_predicted), atol=1e-5)
    assert np.allclose(np.linalg.norm(predicted - transformed, axis = 0), 0, atol = 1e-2)
    

def test_ddl_class_score():
    np.random.seed(10)
    d = DataDrivenLinearization(2, degree = 3)
    True_A = np.array([[-0.001, -0.02], [0.02, -0.001]])
    ic = np.array([1, 1])

    traj = [ic]
    for _ in range(1000):
        traj.append(expm(True_A) @ traj[-1])
    trajectory = [np.array(traj).T]
    d.fit(trajectory)
    score = d.score(trajectory)
    assert score < 1e-4


def test_ddl_class_fit_inverse():
    np.random.seed(10)
    d = DataDrivenLinearization(2, degree = 3)
    True_A = np.array([[-0.001, -0.02], [0.02, -0.001]])
    ic = np.array([1, 1])

    traj = [ic]
    for _ in range(1000):
        traj.append(expm(True_A) @ traj[-1])
    trajectory = [np.array(traj).T]
    d.fit(trajectory)
    transformed = d.transform(trajectory[0])
    d.fit_inverse(trajectory)
    
    inverse_transformed = d.inverse_transform(transformed)

    assert np.allclose(trajectory[0], inverse_transformed, atol=1e-5)



def test_ddl_class_fit_torch():
    np.random.seed(10)
    d = DataDrivenLinearization(2, degree = 3)
    True_A = np.array([[-0.001, -0.02], [0.02, -0.001]])
    ic = np.array([1, 1])

    traj = [ic]
    for _ in range(1000):
        traj.append(expm(True_A) @ traj[-1])
    trajectory = np.array(traj).T
    d.fit_torch(trajectory)
    eigs_true = np.linalg.eig(expm(True_A))[0]
    eigs_predicted = np.linalg.eig(d.linear_model)[0]
    assert np.allclose(np.log(eigs_true), np.log(eigs_predicted), atol=1e-5)


def test_differentiate_model():
    # Define the symbolic variables
    x, y = sy.symbols('x y')

    # Define the function f
    f1 = x - x**2 + x*y
    f2 = y + x**3 - y**2*x
    f = sy.Matrix([f1, f2])

    # Compute the Jacobian
    jacobian = f.jacobian([x, y])
    np.random.seed(10)
    data = np.random.rand(3,2)
    poly = PolynomialFeatures(degree = 3, include_bias=False)
    nonlin_features = poly.fit_transform(data)
    jacobian_exact = sy.lambdify([x,y], jacobian)
    coeff = np.array([[  1,  0, -1,  1,  0,  0,  0,  0,  0],
       [  0,  1,  0, 0,  0, 1,  0,  -1,  0]])
    derivative_mtx = differentiate_model(poly, coeff, data.T)
    assert np.allclose(derivative_mtx, jacobian_exact(data[:,0], data[:,1]))

if __name__ == "__main__":
    test_ddl_class_fit_inverse()
    test_ddl_class_transform_predict()