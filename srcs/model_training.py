import linear_regression.linear_regression	as ln
import linear_regression.vizualisation 		as vz
import linear_regression.params				as params
import numpy 								as np

def main():
    """
    This is the main script that processes the model's training.
    """
    try:
        # Retrieving dataset
        df = ln.retrieve_dataset()
        feature = df[:, 0] # km 
        target = df[:, 1] # price

        # Normalizing matrices shapes
        x = feature.reshape(feature.shape[0], 1)
        y = target.reshape(target.shape[0], 1)

        # Normalizing values
        nx = ln.normalize_val(x)

        # Adding bias column to X
        X = np.hstack((nx, np.ones(nx.shape)))

        # Creating theta vector from params.py attributs
        theta = np.array([params.theta1, params.theta0]).reshape(2, 1)

        # Defining theta
        res = ln.gradient_descent(X, y, theta)
        predictions = ln.model(X, res["theta"])

        # Create 'stats' dir
        vz.mk_statsdir()

        # Plotting dataset
        vz.plot_dataset(x=feature, y=y)

        # Plotting linear regression
        vz.plot_predictions(x=feature, y=y, predictions=predictions)

        # Plotting cost function results
        vz.plot_cost_func(x=range(1, len(res["costs"]) + 1), y=res["costs"])

        # Denormalize theta
        theta  = ln.denormalize_theta(x, res["theta"])

        # Updating model's theta attribut in the params.py
        ln.update_theta(theta)

        # Outputting model's precision
        print(f"model's prediction precision: {float(ln.algo_precision(predictions, y))*100:.2f}%")
    except Exception as e:
        print(type(e).__name__ + ":", e)
    return

if __name__ == "__main__":
    main()