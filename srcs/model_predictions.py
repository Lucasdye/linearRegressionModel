import linear_regression.linear_regression as ln
import linear_regression.params as params
import parsing.parsing as pars
import numpy as np
import signal

def signal_handler(signum, frame):
    """
    Handles the input signals, specifies the where in the code it was received
    and exits 0 the program.
    Arg:
        signum (int): signal code
        frame (frame object): frame from which the signal was caught
    """
    print(f"\nSignal {signum} received at line {frame.f_lineno} in {frame.f_code.co_name}()")
    exit(0)

def waiting_for_input(func):
    """
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
    Outputs the predicted value from the input x.
    Returns:
	0 if success 1 if fail.
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
        print(f"The model's prediction for x = {input} is {res}")
        return 0
    except Exception as e:
        print(type(e).__name__ + ":", e)
        return 1

if __name__ == "__main__":
    main()