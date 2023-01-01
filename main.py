import time
import numpy as np

from init import init_Y
from metrics import X_in_A, distance, prop_alive_func
from stats import sampling_params, metropolis_X_theta

from tqdm import tqdm
import multiprocessing

if __name__ == "__main__":
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
    results = {'Xs': [Xs.copy()], 'Ws': [Ws.copy()], 'thetas': [thetas.copy()], 'epsilon_list': [epsilon_t]}
    start = time.time()

    for _ in range(2): # Updates before the main loop so the Xs are not all the same at the beginning
        for index, X in enumerate(Xs): 
            phi, tau, xi = thetas[index]
            new_X, new_phi, new_tau, new_xi = metropolis_X_theta(X, Y, phi, tau, xi,
                                                                epsilon_t, random_walk_stds)
            Xs[index], thetas[index] = new_X, (new_phi, new_tau, new_xi)

    print(f'Initialisation done in {round(time.time() - start)}s')

    n_max_step = 10000
    pbar = tqdm(range(n_max_step))
    t = 0
    # while epsilon_t > epsilon_final: # Main loop
    for _ in pbar:
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

        # Step 3 : random walk multiprocessed
        indices_oi = [index for index in range(len(Xs)) if Ws[index]>0]

        phis = list(map(lambda tuple : tuple[0], thetas))
        taus = list(map(lambda tuple : tuple[1], thetas))
        xis = list(map(lambda tuple : tuple[2], thetas))

        pool = multiprocessing.Pool(6)
        processes = [pool.apply_async(metropolis_X_theta, args=(Xs[idx], Y, 
                                                                phis[idx], taus[idx], xis[idx],
                                                                epsilon_t, random_walk_stds,)) for idx in indices_oi]

        result = [p.get() for p in processes]

        # update indexes of Xs and thetas that have been considered
        for idx, idx_oi in enumerate(indices_oi):
            Xs[idx_oi] = result[idx][0]
            thetas[idx_oi] = tuple(result[idx][1:])
        
        # save results 
        results['Xs'].append(Xs.copy())
        results['Ws'].append(Ws.copy())
        results['thetas'].append(thetas.copy())
        results['epsilon_list'].append(epsilon_t)

        #update result file
        t+=1
        if t%100==0:
            np.save('results_8.npy', results)

        # print results
        # print(f'loop = {t}; loop time : {round(time.time() - step_start, 3)}s; total_time : {round(time.time() - start)}s')
        pbar.set_description(f'epsilon = {round(epsilon_t, 6)}; epsilon_target = {epsilon_final}')
