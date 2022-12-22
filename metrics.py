import numpy as np

def eta(X): 
    total_pop = np.sum(X)
    eta_1 = len(X)
    eta_2 = 1 - np.sum([(genom_pop/total_pop)**2 for genom_pop in X])
    return eta_1, eta_2

def rho(eta, eta_bar, n):
    '''n is the number of individuals in the population''' # TODO : Laquelle ?
    result =  (1/n * 
               np.abs(eta[0] - eta_bar[0]) - 
               np.abs(eta[1] - eta_bar[1]))
    return result

def X_in_A(Y, X, epsilon):
    eta_1_Y, eta_2_Y = eta(Y)
    eta_1_X, eta_2_X = eta(X)
    pop_in_X = np.sum(X)
    distance = rho((eta_1_X, eta_2_X), (eta_1_Y, eta_2_Y), pop_in_X)
    return distance <= epsilon

def ESS(W):
    return 1/np.sum(W**2)

def prop_alive_func(Ws):
    return np.sum(Ws>0)/len(Ws)