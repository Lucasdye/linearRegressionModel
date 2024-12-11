import sys as system
import numpy as np
from params import params 
from linear_regression import linear_regression as ln
from parsing import parsing as pars
import signal

def signal_handler(signum, frame):
    print(f"\nSignal {signum} received")
    exit(0)

def waiting_for_input(func):
    """
    Summary:
        Blocks the program til the func returns False.
    Returns:
        Returns the return value by func.
    """
    str = ""
    ret = False
    
    try:
        while ret is False:
            str = input("enter value: ")
            ret = func(str)
    except EOFError:
        print("\nEOF error")    
        exit(1)
    return ret


def main():
    """
    Summary:
    Predicts a value of any given mileage in km with a linear regression
    model.
    """
    try:
        # Basic signal handling
        signal.signal(signal.SIGINT, signal_handler)  
        signal.signal(signal.SIGTERM, signal_handler)

        # Reaching and parsing user input
        input = waiting_for_input(pars.is_int_or_float)
        if input["type"] == "int":
            input = int(input["nb"])
        elif input["type"] == "float":
            input = float(input["nb"])

        # Creating X matrix (features with bias column)
        X = np.array(input)
        X = np.hstack((X, np.ones(1)))

        # Creating theta matrix (parameters)
        theta = np.hstack((np.array(params.theta1), np.array(params.theta0))).reshape(2, 1)

        # Result
        res = int(ln.model(X, theta)[0])
        print(f"The model's prediction for x = {input} is {res}$")
    except Exception as e:
        print(type(e).__name__ + ":", e)

if __name__ == "__main__":
    main()