
from linear_regression.params import params 
from matplotlib import pyplot as plt
import os

def mk_statsdir():
    # creating stats dir 
	current_dir = os.getcwd()
	parent_dir = current_dir[:current_dir.rfind("/")]
	new_dir = parent_dir + "/stats"
	os.makedirs(new_dir, exist_ok=True)

def plot_dataset(x, y):
    # Plotting data
    plt.scatter(x, y)
    plt.xlabel(params.x_label)
    plt.ylabel(params.y_label)
    plt.savefig(fname=params.stats_path + "data")
    plt.clf()

def plot_predictions(x, y, predictions):
	# Plotting linear regression
	plt.scatter(x, y)
	plt.xlabel(params.x_label)
	plt.ylabel(params.y_label)
	plt.plot(x, predictions)
	plt.savefig(fname=params.stats_path + "linear_regression")
	plt.clf()

def	plot_cost_func(x, y):
	# Plotting cost function results
	plt.plot(x, y)
	plt.xlabel("epochs")
	plt.ylabel("cost")
	plt.savefig(fname=params.stats_path + "cost_function_res")
	plt.clf()