from linear_regression import linear_regression as ln
import os

# creating stats dir 
current_dir = os.getcwd()
parent_dir = current_dir[:current_dir.rfind("/")]
new_dir = parent_dir + "/stats"
os.makedirs(new_dir, exist_ok=True)

# Calling training algo
ln.training()