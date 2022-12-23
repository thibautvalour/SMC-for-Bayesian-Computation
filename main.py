import time
import numpy as np

from init import init_Y
from metrics import X_in_A, distance, prop_alive_func
from stats import sampling_params, metropolis_X_theta

# Step 0 : initialisation
alpha = 0.9
N = 1000
Nt = 500
Y = init_Y()
Xs = [[1]]*N
epsilon_final = 0.00045
epsilon_t = np.inf
random_walk_stds = (0.05, 0.05, 0.7)
Ws = 1/N * np.ones(N)
prop_alive = prop_alive_func(Ws)
thetas = np.array([sampling_params() for _ in range(N)])
results = {'Xs': [Xs], 'Ws': [Ws], 'thetas': [thetas], 'epsilon_list': [epsilon_t]}
start = time.time()

for _ in range(2): # Updates before the main loop so the Xs are not all the same at the beginning
    for index, X in enumerate(Xs): 
        phi, tau, xi = thetas[index]
        new_X, new_phi, new_tau, new_xi = metropolis_X_theta(X, Y, phi, tau, xi,
                                                             epsilon_t, random_walk_stds)
        Xs[index], thetas[index] = new_X, (new_phi, new_tau, new_xi)

print(f'Initialisation done in {round(time.time() - start)}s')

t = 0
while epsilon_t > epsilon_final: # Main loop
    step_start = time.time()

    # Step 1 : sampling
    distance_list =  np.sort(list(map(distance, Xs, [Y]*N))) # Compute all distance to Y
    epsilon_t = distance_list[int(alpha*len(distance_list)-1)] # Compute new epsilon by finding the alpha qantile of the distance list
    Ws *= np.array([int(X_in_A(Y, X, epsilon_t)) for X in Xs]) # Compute the new weights

    # Step 2 : resampling
    if prop_alive_func(Ws) < 0.5: # Half are dead
        indices = np.random.choice(range(N), size=N, replace=True, p=(Ws/np.sum(Ws)))
        Xs = [Xs[i] for i in indices]
        thetas = [thetas[i] for i in indices]
        Ws = 1/N * np.ones(N)
        
    # Step 3 : random walk
    for index, X in enumerate(Xs):
        if Ws[index] > 0:
            phi, tau, xi = thetas[index]
            new_X, new_phi, new_tau, new_xi = metropolis_X_theta(X, Y, phi, tau, xi,
                                                                 epsilon_t, random_walk_stds)
            Xs[index], thetas[index] = new_X, (new_phi, new_tau, new_xi)

    # save results 
    results['Xs'].append(Xs)
    results['Ws'].append(Ws)
    results['thetas'].append(thetas)
    results['epsilon_list'].append(epsilon_t)
    np.save('results.npy', results)

    # print results
    print(f'loop = {t}; loop time : {round(time.time() - step_start)}s; total_time : {round(time.time() - start)}s')
    print(f'espilon = {round(epsilon_t, 6)}; epsilon_target = {epsilon_final}')

    t+=1