import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
import sys

np.random.seed(42)

budget = 50000
dimension = 10


def s_ES(problem, budget):
    f_opt = sys.float_info.max
    x_opt = None

    # Parameters setting
    mu_ = 5
    lambda_ = 10
    # tau =  1.0 / np.sqrt(problem.meta_data.n_variables)

    # parameters for individual sigma
    tau_p = 1.0 / np.sqrt(2 * problem.meta_data.n_variables)  # Global learning rate
    tau = 1.0 / np.sqrt(2 * np.sqrt(problem.meta_data.n_variables))  # Local learning rate

    
    # Initialization and Evaluation
    # parent,parent_sigma = initialization(mu_,problem.meta_data.n_variables)
    parent,parent_sigma = initialization_individual_sigma(mu_,dimension,problem.meta_data.n_variables)
    parent_f = []
    for i in range(mu_):
        parent_f.append(problem(parent[i]))
        budget = budget - 1
        if parent_f[i] < f_opt:
            f_opt = parent_f[i]
            x_opt = parent[i].copy()

    while problem.state.evaluations < budget:       
        offspring = []
        offspring_sigma = []
        offspring_f = []

        # Recombination
        for i in range(lambda_):
            # o, s = recombination(parent,parent_sigma)
            # o, s = global_intermediate_recombination(parent, parent_sigma)
            # o, s = discrete_recombination(parent,parent_sigma)
            # o, s = discrete_recombination_individual_sigma(parent,parent_sigma)
            # o, s = global_discrete_recombination(parent, parent_sigma)
            o, s = global_discrete_recombination_individual_sigma(parent, parent_sigma)
            offspring.append(o)
            offspring_sigma.append(s)

        # Mutation
        # one_sigma_mutation(offspring,offspring_sigma,tau)
        individual_sigma_mutation(offspring, offspring_sigma, tau_p, tau)

        # Evaluation
        for i in range(lambda_) : 
            offspring_f.append(problem(offspring[i]))
            budget = budget - 1
            if offspring_f[i] < f_opt:
                    f_opt = offspring_f[i]
                    x_opt = offspring[i].copy()
            if (f_opt <= problem.optimum.y) | (budget <= 0):
                break

        # Selection
        # rank = np.argsort(offspring_f)
        # parent = []
        # parent_sigma = []
        # parent_f = []
        # i = 0
        # while ((i < lambda_) & (len(parent) < mu_)):
        #     if (rank[i] < mu_):
        #         parent.append(offspring[i])
        #         parent_f.append(offspring_f[i])
        #         parent_sigma.append(offspring_sigma[i])
        #     i = i + 1

        # (μ+lambda) Selection
        combined = parent + offspring
        combined_f = parent_f + offspring_f
        combined_sigma = parent_sigma + offspring_sigma

        rank = np.argsort(combined_f)

        parent = []
        parent_f = []
        parent_sigma = []

        i = 0
        while (i < len(combined)) and (len(parent) < mu_):
            parent.append(combined[rank[i]])
            parent_f.append(combined_f[rank[i]])
            parent_sigma.append(combined_sigma[rank[i]])
            i += 1
        
    
    print(f_opt,x_opt)
    return f_opt, x_opt


# Initialization
def initialization(mu,dimension,lowerbound = -5.0, upperbound = 5.0):
    parent = []
    parent_sigma = []
    for i in range(mu):
        parent.append(np.random.uniform(low = lowerbound,high = upperbound, size = dimension))
        parent_sigma.append(0.05 * (upperbound - lowerbound))
    return parent,parent_sigma

# Initialization for individual sigma
def initialization_individual_sigma(mu, dimension, lowerbound=-5.0, upperbound=5.0):
    parent = []
    parent_sigma = []
    for i in range(mu):
        parent.append(np.random.uniform(low=lowerbound, high=upperbound, size=dimension))
        parent_sigma.append(np.random.uniform(0.05, 0.1, size=dimension))
    # return np.array(parent), np.array(parent_sigma)
    return parent, parent_sigma

# One-sigma mutation
def one_sigma_mutation(parent, parent_sigma,tau):
    for i in range(len(parent)):
        parent_sigma[i] = parent_sigma[i] * np.exp(np.random.normal(0,tau))
        for j in range(len(parent[i])):
            parent[i][j] = parent[i][j] + np.random.normal(0,parent_sigma[i])
            parent[i][j] = parent[i][j] if parent[i][j] < 5.0 else 5.0
            parent[i][j] = parent[i][j] if parent[i][j] > -5.0 else -5.0  

# Individual sigma mutation
def individual_sigma_mutation(parent, parent_sigma, tau_p, tau):
    g = np.random.normal(0, 1)

    for i in range(len(parent)):  
        for j in range(len(parent[i])):  
            parent_sigma[i][j] = parent_sigma[i][j] * np.exp(tau_p * g + tau * np.random.normal(0, 1))
            parent[i][j] = parent[i][j] + np.random.normal(0, parent_sigma[i][j])
            parent[i][j] = parent[i][j] if parent[i][j] < 5.0 else 5.0
            parent[i][j] = parent[i][j] if parent[i][j] > -5.0 else -5.0          

# Intermediate recombination
def recombination(parent,parent_sigma):
    [p1,p2] = np.random.choice(len(parent),2,replace = False)
    offspring = (parent[p1] + parent[p2])/2
    sigma = (parent_sigma[p1] + parent_sigma[p2])/2 

    return offspring,sigma

# Global intermediate recombination
def global_intermediate_recombination(parent, parent_sigma):
    num_parents = len(parent)
    offspring = np.zeros_like(parent[0])  
    sigma = np.zeros_like(parent_sigma[0])  

    for p in parent:  
        offspring += p  
    for s in parent_sigma:  
        sigma += s  
    
    offspring = offspring / num_parents  
    sigma = sigma / num_parents 

    return offspring, sigma

# global intermediate recombination for individual sigma
def global_intermediate_recombination_individual_sigma(parent, parent_sigma):
    num_parents = len(parent)
    offspring = np.zeros_like(parent[0]) 
    sigma = np.zeros_like(parent_sigma[0])  

    for p in parent:  
        offspring += p  
    for s in parent_sigma: 
        sigma += s  
    
    offspring = offspring / num_parents  
    sigma = sigma / num_parents 

    return offspring, sigma

# discrete recombination
def discrete_recombination(parent, parent_sigma):
    [p1, p2] = np.random.choice(len(parent), 2, replace=False)
    offspring = []  
    sigma = []  
    
    for i in range(len(parent[p1])): 
        offspring.append(np.random.choice([parent[p1][i], parent[p2][i]])) 
    
    sigma = np.random.choice([parent_sigma[p1], parent_sigma[p2]])  
    return offspring, sigma

# discrete recombination for individual sigma
def discrete_recombination_individual_sigma(parent, parent_sigma):
    [p1, p2] = np.random.choice(len(parent), 2, replace=False)
    offspring = []  
    sigma = []  
    
    for i in range(len(parent[p1])): 
        offspring.append(np.random.choice([parent[p1][i], parent[p2][i]])) 
        sigma.append(np.random.choice([parent_sigma[p1][i], parent_sigma[p2][i]]))

    return offspring, sigma


# global discrete recombination
def global_discrete_recombination(parent, parent_sigma):
    offspring = []  
    sigma = []  

    for i in range(len(parent[0])): 
        offspring.append(np.random.choice([parent[p][i] for p in range(len(parent))])) 
    sigma = np.random.choice(parent_sigma)  
    return offspring, sigma

# global discrete recombination for individual sigma
def global_discrete_recombination_individual_sigma(parent, parent_sigma):
    offspring = []  
    sigma = []  

    for i in range(len(parent[0])):  
        offspring.append(np.random.choice([parent[p][i] for p in range(len(parent))])) 
        sigma.append(np.random.choice([parent_sigma[p][i] for p in range(len(parent))])) 

    return offspring, sigma


def create_problem(fid: int):
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.BBOB)

    l = logger.Analyzer(
        root="data",  
        folder_name="run",  
        algorithm_name="globdiscre_indiv_mu+lamb",  
        algorithm_info="Practical assignment part2 of the EA course",
    )
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    F23, _logger = create_problem(23)
    for run in range(20): 
        f_opt, x_opt = s_ES(F23, budget)
        F23.reset() 
    _logger.close() 

