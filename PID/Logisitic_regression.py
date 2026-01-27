import numpy as np
import math

def sigmoid(z):

    return 1/(1+np.exp(-z))

def rescale(X):

    means = np.mean(X,axis=0)
    stds = np.std(X,axis=0)

    return (X-means)/stds, means, stds

def compute_cost(X, y, w, b,lambda_):

    if X.ndim == 1:
        X = X[:, None]

    m,_ = X.shape
    z = np.dot(X, w)+b

    f = sigmoid(z)    
    total_cost =  np.sum(-y*(np.log(f)) - (1-y)*(np.log(1-f)))/m +np.sum(w**2)*(lambda_)/(2*m)

    return total_cost

def get_cost_der(X,Y,w,b):

    if X.ndim == 1:
        X = X[:, None]

    m,_ = X.shape
    z = np.dot(X,w)+b

    f = sigmoid(z)

    err = f-Y

    dJ_dw = np.dot(X.T,err)/m

    dJ_db = np.sum(err)/m

    return dJ_dw, dJ_db


def log_reg(X,Y,w_in,b_in,N_it,alpha=0.1,gamma=0):
    
    if X.ndim == 1:
        X = X[:, None]

    m,n = X.shape

    w = w_in
    b = b_in

    J_history = []
    w_history = []

    for i in range(N_it):

        dJ_dw, dJ_db = get_cost_der(X,Y,w,b)

        dJ_dw = dJ_dw + (gamma/m)*w

        w = w - alpha*dJ_dw
        b = b - alpha*dJ_db

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  compute_cost(X, Y, w, b,gamma)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(N_it/10) == 0 or i == (N_it-1):
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):.2e}, w = {w},  dJ_dw = {dJ_dw} ")



    return w,b,J_history, w_history

def prediction(X,w,b,th=0.5):
    if X.ndim == 1:
        X = X[:, None]
    z = np.dot(X,w)+b

    f = sigmoid(z)

    return (f > th).astype(float)
