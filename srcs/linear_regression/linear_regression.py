# Modules
import sys
from	linear_regression.params import params
# import  linear_regression.vizualisation as vz
import 	numpy	as np
import	pandas	as pd
import	numpy	as np
import	matplotlib.pyplot as plt

def algo_precision(pred, y):
	"""
	Calculates the precision of the overhole predictions.
	By precision we understand the capacity of the model
	to explain the current dataset variations.

	Args:
		pred (list of int or float): model's predictions
		y (list of int or float): dataset targets

	Returns:
		returns the precision of the model
	"""
	u = ((y - pred)**2).sum() # Residual Sum of Squares (RSS)
	v = ((y - y.mean())**2).sum() # Total Sum of Squares (TSS)
	return 1 - (u/v) # 1 - (unexplained variation / total variation)

def model(x, theta):
    """
    This is the model's equation from which the predictions are made.
    It multiplies the x matrix by the theta matrix. 

	Args:
		x (numpy array of int or floats): matrix shape (n, 2), column x and bias column.
		theta (numpy array of thetas): matrix shape (2, 1), parameter theta0 and theta1.

	Returns:
		returns a matrix of all predictions.
    """
    return x.dot(theta)

def cost(x, y, theta):
    """
    This function measures the MSE (mean square error) of the model. 

    Args:
        x (numpy array of int or floats): matrix shape (n, 2), column x and bias column.
        y (numpy array of int or floats): matrix shape (n, 1), column of targets
        theta (numpy array of thetas): matrix shape (2, 1), parameter theta0 and theta1.

	Returns:
		returns a matrix of all predictions.
	"""
    m = len(y)
    return (np.sum((model(x, theta) - y) ** 2)) / (2 * m)

def gradient(x, y, theta):
    """
    This function computes the gradients for
    the current theta0 and theta1 values. 

    Args:
        x (numpy array of int or floats): matrix shape (n, 2), column x and bias column.
        y (numpy array of int or floats): matrix shape (n, 1), column of targets.
        theta (numpy array of thetas): matrix shape (2, 1), parameter theta0 and theta1.

    Returns:
        Returns the gradients of theta0 and theta1 in a matrix
    """
    m = len(y)
    return 1/m * x.T.dot(model(x, theta) - y)

def gradient_descent(x, y, theta):
    """
    This function performs the gradient descent algorithm.
    The main idea is to compute new theta values based upon
    their gradient values. If the gradient is negative,
    we increase the theta value and vice versa.
    The increasing or diminishing of the thetas is
    tamed by the learning rate and updated times the epoch value.
    
    Args:
        x (numpy array of int or floats): matrix shape (n, 2), column x and bias column.
        y (numpy array of int or floats): matrix shape (n, 1), column of targets.
        theta (numpy array of thetas): matrix shape (2, 1), parameter theta0 and theta1.
        params.lr (float): learning rate.
        params.epoch (int): how many time thetas are updated.

    Returns:
        Returns the ideal theta0 and theta1 values and the mse for each iteration.
    """
    lst_cost = []
    for i in range(0, params.epoch):
        theta = theta - params.lr * gradient(x, y, theta)
        lst_cost.append(float(cost(x, y, theta)))
    return {"theta":theta, "costs":lst_cost}

def retrieve_dataset():
    """
    Retrieves the dataset specified by the dataset_path attribut
    defined in the params/params.py file.
    
    Returns:
        Returns the dataset in the form of a numpy matrix
    """
    # Retrieving dataset and converting to numpy array
    df = pd.read_csv(params.dataset_path)
    df = df.to_numpy()
    return df

def normalize_val(x):
    """
    Normalizes x values between 0 and 1. Very useful to prevent float overflow
    for big feature values.

    Args:
        x (numpy matrix): values to be normalized.
    Returns:
        Returns a numpy matrix of nomalized values.
    """
    nx = np.zeros(len(x), dtype=float).reshape(len(x), -1)
    xmin = x.min()
    xmax = x.max()
    for i in range(0, len(x)):
        nx[i] = (x[i] - xmin) / (xmax - xmin)
    return nx
    
def denormalize_theta(x, theta):
    """
    Denormalizes theta to get real theta values for predictions.

    Args:
        x (numpy matrix): values from which we will denormalize.
    Returns:
        Returns a numpy matrix  denormalized theta values.
    """
    xmin = x.min()
    xmax = x.max()
    
    # Renormalize theta1 (slope)
    theta[0] = theta[0] / (xmax - xmin)
    # Renormalize theta0 (intercept)
    theta[1] = theta[1] - (theta[0] * xmin) / (xmax - xmin)
    return theta

def update_theta(theta):
    """
    Updates the theta0 and theta1 attributes in the params.py

    Args:
        theta (numpy matrix): shape(n, 1)
    """
    with open("linear_regression/params/params.py", "r") as f:
        content = f.read()
        content = content.replace("theta0 = 0", f"theta0 = {float(theta[1])}")
        content = content.replace("theta1 = 0", f"theta1 = {float(theta[0])}")
    with open("linear_regression/params/params.py", "w") as f:
        f.write(content)
 

