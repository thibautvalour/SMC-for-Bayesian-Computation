import numpy as np
import scipy as scipy
from scipy import stats
from metrics import X_in_A

def sampling_params():
    phi = np.random.gamma(1, 10) # new_infection_rate
    xi = scipy.stats.truncnorm.rvs(a=0, b=np.inf,
                                   loc=0.198, scale=0.06735) # TODO : vérifier la moyenne  # death_rate
    tau = np.random.uniform(low=0.0, high=phi) # mutation_rate
    return phi, tau, xi

def random_walk_on_params(phi, tau, xi, stds):
    while True:
        new_phi = np.random.normal(phi, stds[0])
        new_tau = np.random.normal(tau, stds[1])
        new_xi = np.random.normal(xi, stds[2])
        if new_phi > 0 and new_tau > 0 and new_xi > 0 and new_tau < new_phi:
            return new_phi, new_tau, new_xi

def pi_n(phi, tau, xi):
    phi_density = stats.gamma.pdf(phi, 1, scale=10)
    tau_density = stats.uniform.pdf(tau, 0, phi)
    xi_density = stats.truncnorm.pdf(xi, a=0, b=np.inf, loc=0.198, scale=0.06735)
    theta_density = phi_density * tau_density * xi_density
    return theta_density

def X_update(X, phi, tau, xi):
    total_pop = np.sum(X)
    new_X = X.copy()
    genom_affected = np.random.choice(range(len(X)),
                                      p=[genom_pop/total_pop for genom_pop in X])
    normalize_coeff = phi + tau + xi
    outcome = np.random.choice(['birth', 'death', 'mutation'],
                                p=[phi/normalize_coeff, tau/normalize_coeff,
                                   xi/normalize_coeff])
    if outcome == 'birth':
        new_X[genom_affected] += 1
    elif outcome == 'death':
        new_X[genom_affected] -= 1
    else: # Mutation
        new_X[genom_affected] -= 1
        new_X.append(1)
    return new_X

def metropolis_X_theta(X, Y, phi, tau, xi, epsilon_t, random_walk_stds):
    theta_density = pi_n(phi, tau, xi)
    while True: # Parameters are sampled until they are accepted
        new_phi, new_tau, new_xi  = random_walk_on_params(phi, tau, xi, random_walk_stds)
        new_theta_density = pi_n(new_phi, new_tau, new_xi)
        new_X = X_update(X, new_phi, new_tau, new_xi)
        acceptance_proba = min(1, X_in_A(Y, new_X, epsilon_t)*new_theta_density/theta_density)

        if acceptance_proba >= np.random.uniform(): 
            return new_X, new_phi, new_tau, new_xi