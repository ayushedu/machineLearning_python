import numpy as np

def computeCost(X,y,theta):
    """
    Compute cost for linear regression
    """
    m = len(y) # No of training examples

    # Compute cost
    J = 1/(2*m) * np.sum(np.power(np.subtract(np.dot(X.values,theta),y.values),2))
    
    return J
