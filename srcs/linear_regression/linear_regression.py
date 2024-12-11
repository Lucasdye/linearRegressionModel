# Modules
import sys
# Add the directory where params.py is located
sys.path.append("/home/lu/Coding/ft_linear_regression_git/srcs")
from	params	import params
import 	pandas	as pd
import 	numpy	as np
import	pandas	as pd
import	numpy	as np
import	matplotlib.pyplot as plt

def algo_precision(pred, y):
	u = ((y - pred)**2).sum()
	v = ((y - y.mean())**2).sum()
	return 1 - (u/v)

def model(x, theta):
	return x.dot(theta)

def cost(x, y, theta):
	m = len(y)
	return (np.sum((model(x, theta) - y) ** 2)) / (2 * m)

def gradient(x, y, theta):
	m = len(y)
	return 1/m * x.T.dot(model(x, theta) - y)
	# return x.T.dot((model(x, theta) - y)) / (1 / m)

def gradient_descent(x, y, theta, lr, turns):
	lst_cost = []
	for i in range(0, turns):
		theta = theta - lr * gradient(x, y, theta)
		lst_cost.append(float(cost(x, y, theta)))
	return {"theta":theta, "costs":lst_cost}

def retrieve_dataset(path):
	# Retrieving dataset and converting to numpy array
	df = pd.read_csv(path)
	df = df.to_numpy()
	return df

def training():

	try:
		# Dataset
		df = retrieve_dataset(params.data_path)
		feature = df[:, 0] # km 
		target = df[:, 1] # price

		# Normalizing matrices shapes
		x = feature.reshape(feature.shape[0], 1)
		y = target.reshape(target.shape[0], 1)
		nx = np.zeros(len(x), dtype=float).reshape(-1, 1)

		# Normalizing values between 0 and 1 (prevents float underflow)
		xmin = np.min(x)
		xmax = np.max(x)
		for i in range(len(x)):
			nx[i] = (x[i] - xmin) / (xmax - xmin)

		# Adding bias column to X
		X = np.hstack((nx, np.ones(nx.shape)))
	
		# Creating theta vector from random values
		theta = np.array([params.theta1, params.theta0]).reshape(2, 1)

		# Defining theta
		res = gradient_descent(X, y, theta, lr=params.lr, turns=params.turns)
		predictions = model(X, res["theta"])

		# Plotting data
		plt.scatter(feature, y)
		plt.xlabel("Mileage (km)")
		plt.ylabel("Price")
		plt.savefig(fname=params.stats_path + "data")
		plt.clf()

		# Plotting linear regression
		plt.scatter(feature, y)
		plt.xlabel("Mileage (km)")
		plt.ylabel("Price")
		plt.plot(feature, predictions)
		plt.savefig(fname=params.stats_path + "linear_regression")
		plt.clf()

		# Plotting cost function results
		plt.plot(range(1, len(res["costs"]) + 1) ,res["costs"])
		plt.xlabel("turns")
		plt.ylabel("cost")
		plt.savefig(fname=params.stats_path + "cost_function_res")
		plt.clf()

		# Renormalize theta1 (slope)
		theta1_real = res["theta"][0] / (xmax - xmin)

		# Renormalize theta0 (intercept)
		theta0_real = res["theta"][1] - (res["theta"][0] * xmin) / (xmax - xmin)
	
		# Write down the new theta in params.py
		with open("./params/params.py", "r") as f:
			content = f.read()
			content = content.replace("theta0 = 0", f"theta0 = {float(theta0_real)}")
			content = content.replace("theta1 = 0", f"theta1 = {float(theta1_real)}")
			with open("./params/params.py", "w") as f:
				f.write(content)

		# Output algo precision
		print(f"model's prediction precision: {float(algo_precision(predictions, y))*100:.2f}%")
	except Exception as e:
		print(type(e).__name__ + ":", e)
		return 

