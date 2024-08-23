import numpy as np
import scipy.stats as scs
import emcee
import matplotlib.pyplot as plt
import seaborn as sns
import math
from math import sqrt, log, cos, pi, exp
import os
from pyvbmc import VBMC
from matplotlib import ticker
import corner
from math import cos, sin, pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector
from abcpy.continuousmodels import Uniform
from abcpy.statistics import Identity
from abcpy.distances import Euclidean
from abcpy.inferences import RejectionABC
from abcpy.backends import BackendDummy as Backend
from scipy.integrate import solve_ivp
import jax.numpy as jnp
import blackjax
import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt
import seaborn as sns
import os
import corner
from math import sqrt, log, cos, pi, exp
from scipy.integrate import solve_ivp
import jax
import jax.numpy as jnp
import blackjax
import jax.numpy as jnp
from jax.scipy.stats import norm
from scipy.stats import gaussian_kde

from abcpy.backends import BackendDummy as Backend
from abcpy.continuousmodels import Uniform, Normal
from abcpy.statistics import Identity
from abcpy.statisticslearning import Semiautomatic, SemiautomaticNN
from abcpy.distances import Euclidean
from abcpy.perturbationkernel import DefaultKernel
from abcpy.inferences import PMCABC
import numpy as np
import logging
from scipy.stats import gaussian_kde



class SimulationModel(ProbabilisticModel, Continuous):
    def __init__(self, parameters, ffun, name='SimulationModel'):
        if not isinstance(parameters, list):
            raise TypeError('Input of SimulationModel is of type list')
        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)
        self.ffun = ffun
        self.results=[]

    def _check_input(self, input_values):

        return all([value >= 0 for value in input_values])

    def _check_output(self, values):
        if not isinstance(values, np.array):
            raise ValueError('This returns an array')
        
        if values.shape[0] != len(self.ffun): 
            raise RuntimeError(f'The size of the output has to be {len(self.ffun)}.')
        
        return True

    def get_output_dimension(self):
        return len(self.ffun)

    def forward_simulate(self, input_values, k, rng=np.random.RandomState()):
        simulation_result = self.ffun(input_values)
        result = [simulation_result for _ in range(k)]
        if len(result)>2001:
            result=np.array(result)[::int(len(result)/1000)]
        print(len(result))  
        result=np.array(result)
        result=result.flatten().tolist()
        return result

        

class Bayesian_inference:
    def __init__(self, kwargs, input_data,ffun,adjusted_params,algorithm=""):
        self.boundaries=kwargs['boundaries']
        self.lower_bound = kwargs['boundaries'][0]
        self.upper_bound = kwargs['boundaries'][1]
        self.D = kwargs['D']
        self.renormalized = []
        self.lamb = kwargs['lamb']
        self.sigma = kwargs['sigma']
        self.noise = kwargs['noise']
        self.prior_mean = kwargs['prior_mean']
        self.prior_std = kwargs['prior_std']
        self.input_trace = input_data
        self.dt = kwargs['dt']  
        self.incovariance = []
        self.tvec = []
        self.ffun=ffun
        self.adjusted_params=adjusted_params
        self.labels=["1","2","3"]
        self.map=[]
        self.observed=self.input_trace
        self.algorithm=algorithm
        if "LIKELIHOOD_FREE" in self.algorithm :
            print("yes")
            self.model=self.create_model()   
    def resample(self, input_trace):
        if len(input_trace) > 2001:
            self.x = int(len(input_trace) / 1000)
        else:
            self.x = 1  
        self.renormalized = input_trace[::self.x]
        self.recompute_dt()  
        return self.renormalized
    def labelser(self):
        labels=[]
        for i in self.adjusted_params:
            p=i.strip().split()
            labels.append(p[len(p)-1])
        self.labels=labels
        return self.labels   

    def recompute_dt(self):
        self.new_dt = self.dt * self.x
        return self.new_dt

    def aut_corr_func(self, y, D, lamb):
        return lamb * D * np.exp(-lamb * np.abs(y))

    def cov_mat(self, f, t_vec):
        return np.array([[f(abs(t_vec[t1] - t_vec[t2]), self.D, self.lamb) for t2 in range(len(t_vec))] for t1 in range(len(t_vec))])

    def inv_cov_mat(self, f, t_vec, regularization=1e-8):
        covmat = self.cov_mat(f, t_vec)
        covmat += regularization * np.eye(covmat.shape[0])  
        try:
            invcovmat = np.linalg.inv(covmat)
        except np.linalg.LinAlgError:
            print("Covariance matrix is singular, adding more regularization.")
            covmat += regularization * 10 * np.eye(covmat.shape[0])
            invcovmat = np.linalg.inv(covmat)
        self.incovariance = invcovmat

        return invcovmat

    def diagonal_cov_mat(self, sigma, length=None):
        if length is None:
            length = len(self.renormalized)
        d = 1 / (sigma ** 2)
        inv_covmat = np.zeros((length, length), dtype=float)
        np.fill_diagonal(inv_covmat, d)
        self.incovariance = inv_covmat
        return inv_covmat

    def generate_time_vector(self, duration):
        t_vec = np.arange(0, duration + self.new_dt, self.new_dt)
        self.tvec = t_vec
        return t_vec

    def compute_covariance(self):
        if self.noise == 'colour':
            t_vec = self.tvec
            return self.inv_cov_mat(self.aut_corr_func, t_vec)
        else:
            return self.diagonal_cov_mat(self.sigma, len(self.renormalized))
            
    def log_prior(self, theta):
        """Independent normal prior."""
        if "NEW_HMC" in self.algorithm:
            if all(self.lower_bound[i] < theta[i] < self.upper_bound[i] for i in range(len(theta))):
                 return jnp.sum(norm.logpdf(theta, np.array(self.prior_mean), np.array(self.prior_std)))
        
        if all(self.lower_bound[i] < theta[i] < self.upper_bound[i] for i in range(len(theta))):
            return np.sum(scs.norm.logpdf(theta, self.prior_mean, self.prior_std))
        return -np.inf
        
    def initial_candidate_solution(self):
        return self.ffun([0.01, 0.01, 0.0001])
        
    def log_likelihood( self,candidates):
        log_likelihood_sum = 0
        differences = self.ffun(candidates)
        new_differences=differences[::self.x]
        diff_t=new_differences.T
        log_likelihood_sum += np.sum(-0.5 * diff_t @ self.incovariance @ new_differences)
        return log_likelihood_sum
    
    def log_probability(self,candidates):
        lp = self.log_prior(candidates)
        if not np.isfinite(lp):
             return -np.inf
             
        return lp + self.log_likelihood(candidates) 
        
    def run_mcmc_and_plot_samples(self,base_dir,ndim, nwalkers, initial_pos, nsteps=200, discard=20, thin=5):
        ndim=int(ndim)
        nwalkers=int(nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)
        print(ndim)
        print(type(initial_pos))
        print(initial_pos)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)
        print("done till here")
        samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
        log_probs = np.array([self.log_probability(sample) for sample in samples])
        print("done upt")
        map_estimate = samples[np.argmax(log_probs)]
        self.map=map_estimate
        print("MAP estimate:", map_estimate)
        print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
        return samples
    def run_custom_variational_inference(self, base_dir, x0, regularization=3e-4):
        bounds = np.array([self.lower_bound, self.upper_bound]).T

      # Define the variational family (e.g., Gaussian)
        variational_mean = np.copy(x0)
        variational_std = np.ones_like(x0) * self.sigma

        # Custom optimization loop to update variational parameters
        for _ in range(3):  # Number of iterations
            # Ensure the std is non-negative
            print(f"Iteration {_+1} of 3")
            variational_std = np.maximum(variational_std, 1e-6)
    
            # Sample from the variational distribution
            samples = np.random.normal(variational_mean, variational_std, size=(10, len(x0)))
    
            # Compute the log-likelihood and log-prior for each sample
            log_likelihoods = np.array([self.log_likelihood(s) for s in samples])
            log_priors = np.array([self.log_prior(s) for s in samples])
    
            # Compute the evidence lower bound (ELBO)
            elbo = np.mean(log_likelihoods + log_priors)
    
            # Update the variational parameters (mean and std)
            grad_mean = np.mean(samples - variational_mean, axis=0)
            grad_std = np.mean((samples - variational_mean) ** 2 / variational_std - 1, axis=0)
            print(grad_mean)
            print(grad_std)
    
            variational_mean += 0.01 * grad_mean  # Gradient ascent step
            variational_std += 0.01 * grad_std  # Gradient ascent step
    
        # After optimization, ensure std is non-negative
        variational_std = np.maximum(variational_std, 1e-6)
        
        # Sample from the optimized variational distribution
        final_samples = np.random.normal(variational_mean, variational_std, size=(int(100), len(x0)))
    
        # Select the MAP estimate (sample with the highest posterior probability)
        log_probs = np.apply_along_axis(self.log_probability, 1, final_samples)
        max_log_prob_index = np.argmax(log_probs)
        map_estimate = final_samples[max_log_prob_index]
        self.map = map_estimate
    
        return final_samples
    
        
        
    def plotting(self,ndim,samples,base_dir):
        ndim=int(ndim)
        fig, axes = plt.subplots(1, ndim, figsize=(5 * ndim, 5))
        for i in range(ndim):
             sns.histplot(samples[:, i], bins=50, kde=True, ax=axes[i])
             axes[i].set_title(f'Distribution of ${self.labels[i]}$')
             axes[i].set_xlabel(self.labels[i])
             axes[i].set_ylabel('Density')
        plt.tight_layout()
    
        fig_path = os.path.join(base_dir, "figures.png")
        plt.savefig(fig_path)
    
        plt.tight_layout()
        #plt.show()
    def run_variational_inference(self,base_dir,x0,regularization=3e-4):
        bounds = np.array([self.lower_bound, self.upper_bound]).T 
        def regularized_log_probability(theta):
            try:
                return self.log_probability(theta)
            except np.linalg.LinAlgError:
            # Regularize by adding a small noise
                return self.log_probability(theta + np.random.normal(0, regularization, theta.shape))
        vbmc = VBMC(regularized_log_probability, x0, bounds[:, 0], bounds[:, 1])
        try:
            vp, results = vbmc.optimize()
        except np.linalg.LinAlgError as e:
            print("Encountered a linear algebra error during optimization:", e)
            return None, None
        
        vbmc = VBMC(regularized_log_probability, x0, bounds[:, 0], bounds[:, 1])
        vp, results = vbmc.optimize()
        n_samples = int(30e5)
        Xs, _ = vp.sample(n_samples)
        print(len(Xs))
        print(len(Xs[-1]))
        print(Xs[-1])        
        #post_mean = np.mean(Xs, axis=0)
        #post_cov = np.cov(Xs.T)
        

        subsampled_Xs = Xs[::100000]
        log_probs = np.apply_along_axis(self.log_probability, 1, subsampled_Xs)
        max_log_prob_index = np.argmax(log_probs)        
        map_estimate = subsampled_Xs[max_log_prob_index]
        self.map=map_estimate
        print("MAP estimate:", map_estimate)
        return Xs
    
        
    def plotvi(self,Xs,base_dir):
    
        print("works till plotting")
        param_labels = self.labels
    
        fig = corner.corner(
             Xs,
             labels=param_labels,
             color=None,
             show_titles=True,
             title_fmt='.5f',
             fill_contours=False,
             plot_datapoints=False,
             plot_density=False,
             contour_kwargs={
                'levels': 8, 
                'locator': ticker.MaxNLocator(prune='lower'), 
                'colors': None, 
                'cmap': 'rainbow', 
                'linewidths': 1.5
             }
        )
        axes = np.array(fig.axes).reshape((len(param_labels), len(param_labels)))
        for ax in axes.flatten():
             ax.grid(True)
        file_path = os.path.join(base_dir, 'figures.png')
        plt.savefig(file_path)         
        #plt.show()
    def run_hmc(self, n_samples, step_size, num_steps):
        def potential_fn(params):
            return -self.log_probability(params)

        def grad_potential_fn(params):
            return jax.grad(potential_fn)(params)

        rng_key = jax.random.PRNGKey(0)
        lower_bound_jax = jnp.array(self.lower_bound)
        upper_bound_jax = jnp.array(self.upper_bound)
        init_params = jax.random.uniform(rng_key, shape=(len(lower_bound_jax),), minval=lower_bound_jax, maxval=upper_bound_jax)
        hmc = blackjax.hmc(potential_fn, grad_potential_fn, step_size, num_steps)
        hmc_state = hmc.init(init_params)

        @jax.jit
        def one_step(state, rng_key):
            state, info = hmc.step(state, rng_key)
            return state, state.position

        samples = []
        for _ in range(n_samples):
            rng_key, subkey = jax.random.split(rng_key)
            hmc_state, sample = one_step(hmc_state, subkey)
            samples.append(sample)

        return jnp.array(samples)     
        
        
    def hamiltonian(self, q, p):
        return -self.log_probability(q) + 0.5 * np.sum(p**2)

    def hamilton_equations(self, t, y):
        q, p = y[:len(self.lower_bound)], y[len(self.lower_bound):]
        dqdt = p
        dpdt = -self.grad_log_probability(q)
        return np.concatenate([dqdt, dpdt])

    def grad_log_probability(self, q):
        # Compute gradient numerically
        eps = 1e-8
        grad = np.zeros_like(q)
        for i in range(len(q)):
            q_plus = q.copy()
            q_plus[i] += eps
            q_minus = q.copy()
            q_minus[i] -= eps
            print("self.log_probability(q_plus):",self.log_probability(q_plus))
            grad[i] = (self.log_probability(q_plus) - self.log_probability(q_minus)) / (2 * eps)
            #print(f"Gradient component {i}: {grad[i]}")
        return grad

    def leapfrog_step(self, q, p, epsilon):
        p -= 0.5 * epsilon * self.grad_log_probability(q)
        q += epsilon * p
        p -= 0.5 * epsilon * self.grad_log_probability(q)
        return q, p

    def run_hmc1(self, n_samples, epsilon, L):
        q = np.random.uniform(self.lower_bound, self.upper_bound)
        samples = []
        
        for _ in range(n_samples):
            q0 = q
            p0 = np.random.normal(0, 1, len(q))
            
            q, p = q0, p0
            for _ in range(L):
                q, p = self.leapfrog_step(q, p, epsilon)
            
            p = -p
            
            current_H = self.hamiltonian(q0, p0)
            proposed_H = self.hamiltonian(q, p)
            
            if np.random.random() < np.exp(current_H - proposed_H):
                samples.append(q)
            else:
                samples.append(q0)
                q = q0
        
        return np.array(samples)        
        
        
    def create_model(self):
        from abcpy.continuousmodels import Uniform
        parameters = []
        for i, (lower, upper, label) in enumerate(zip(self.lower_bound, self.upper_bound, self.labels)):
             parameters.append(Uniform([[float(lower)],[float(upper)]], name=f'param{i+1}'))
        print(parameters)


        simulation_model = SimulationModel(parameters, self.ffun)
        
        return simulation_model
        
        
        
    def find_mode_kde(self, values):
        # Transpose values to ensure each column is a variable (dimension)
        values = np.transpose(values)
        kde = gaussian_kde(values)
        x_grid = np.linspace(np.min(values), np.max(values), 1000)
        kde_values = kde(x_grid)
        modes = x_grid[np.argwhere(kde_values == np.max(kde_values)).flatten()]
        return modes[0]  # Return the first occurrence if multiple modes are found


    def run_inference(self, n_samples=2,step_size=0.01, num_steps=10,n_samples_per_param=1, epsilon=10000000,L=10):
        if "NEW_HMC" in self.algorithm:
            samples = self.run_hmc(n_samples, step_size, num_steps)
            print(samples)
            return samples
    
        if  "HMC" in self.algorithm:
             epsilon = epsilon
             L = L
             samples = self.run_hmc1(n_samples, epsilon, L)
             print(samples)
             return samples
        else:
             backend = Backend()
        
             statistics_calculator = Identity()
             distance_calculator = Euclidean(statistics_calculator)
        
        
             print(len(self.input_trace))
             sampler = RejectionABC([self.model], [distance_calculator], backend, seed=42)
             print("First few values of observed data np.array(self.observed[0][:5]) are :", np.array(self.observed[0][:5]))
             print("Observed data shape:", np.array(self.input_trace).shape)
             print("type of observed  is",type(np.array(self.input_trace)))
             input_new=np.array(self.renormalized)
             observed_data = input_new.flatten().tolist()
             #distance = distance_calculator.distance(observed_data,new_new)
             #print("distance is",distance)
             #print("working")
             journal = sampler.sample([observed_data], n_samples, n_samples_per_param, epsilon)
             #print("working still")
             self.journal=journal
             parameters = journal.get_accepted_parameters()
             params_array = np.array(parameters)
             self.map = [self.find_mode_kde(params_array[:, i]) for i in range(params_array.shape[1])]
             print(self.map)
             return journal
             
    def run_pmc(self, num_iterations, num_particles, initial_proposal_std, proposal_std_decay):
        # Set up logging
        logging.basicConfig(level=logging.INFO)

        # Define backend
        backend = Backend()

        # Define prior distributions
        mu = Uniform([[150], [200]], name="mu")
        sigma = Uniform([[5], [25]], name="sigma")

        # Define the model
        height = Normal([mu, sigma])

        # Define statistics
        statistics_calculator = Identity(degree=3, cross=True)

        # Learn the optimal summary statistics using Semiautomatic
        statistics_learning = Semiautomatic([height], statistics_calculator, backend,
                                            n_samples=1000, n_samples_per_param=1, seed=1)
        new_statistics_calculator = statistics_learning.get_statistics()

        # Learn the optimal summary statistics using SemiautomaticNN
        statistics_learning = SemiautomaticNN([height], statistics_calculator, backend,
                                              n_samples=1000, n_samples_val=200, n_epochs=20, use_tqdm=False,
                                              n_samples_per_param=1, seed=1, early_stopping=True)
        new_statistics_calculator = statistics_learning.get_statistics()

        # Define distance
        distance_calculator = Euclidean(new_statistics_calculator)

        # Define kernel
        kernel = DefaultKernel([mu, sigma])

        # Define sampler
        sampler = PMCABC([height], [distance_calculator], backend, kernel, seed=1)

        eps_arr = np.array([500])  # Starting value of epsilon; the smaller, the slower the algorithm.
        n_samples_per_param=1
        epsilon_percentile = 10
        journal = sampler.sample([self.observed], num_iterations, eps_arr, num_particles, n_samples_per_param, epsilon_percentile)

        self.journal = journal
        parameters = journal.get_accepted_parameters()
        params_array = np.array(parameters)
        self.map = [self.find_mode_kde(params_array[:, i]) for i in range(params_array.shape[1])]
        print(self.map)
        return journal

    def find_mode_kde(self, data):
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x = np.linspace(min(data), max(data), 1000)
        kde_pdf = kde(x)
        return x[np.argmax(kde_pdf)]
        
        

    def plot_results(self, journal,base_dir):
        parameters = journal.get_accepted_parameters()
        fig, axs = plt.subplots(len(parameters[0]), 1, figsize=(10, 6))
        
        for i, param in enumerate(parameters[0]):
            axs[i].hist([param[i] for param in parameters], bins=30, density=True)
            axs[i].set_title(f"Posterior of {self.labels[i]}")
        
        print("issue here")
        plt.tight_layout()
        plot_path = os.path.join(base_dir, "figures.png")
        plt.savefig(plot_path)
        #plt.show()
        
    def map_estimate(self):
        parameters = self.journal.get_accepted_parameters()
        params_array = np.array(parameters)
        map_estimates = [np.argmax(np.bincount(params_array[:, i])) for i in range(params_array.shape[1])]
        
        self.map = map_estimates  # Store MAP estimates in self.map

        print("MAP Estimates:")
        for i, param in enumerate(map_estimates):
            print(f"{self.labels[i]}: {param}")
        return map_estimates
        
    def plot_hmc_results(self, samples, base_dir):
        ndim = samples.shape[1]
        fig, axes = plt.subplots(1, ndim, figsize=(5 * ndim, 5))
        for i in range(ndim):
            sns.histplot(samples[:, i], bins=50, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of ${self.labels[i]}$')
            axes[i].set_xlabel(self.labels[i])
            axes[i].set_ylabel('Density')
        plt.tight_layout()
    
        fig_path = os.path.join(base_dir, "hmc_figures.png")
        plt.savefig(fig_path)
        plt.show()    
        
     
        
    
          
    
 
        
   
