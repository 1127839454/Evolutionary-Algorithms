from typing import Tuple
import numpy as np
import sys
sys.path.append('/path/to/ioh')  # Replace with the actual path to the ioh package
import ioh
from ioh import get_problem, logger, ProblemClass

budget = 5000

np.random.seed(42)

def s_GA(problem: ioh.problem.PBO, population_size, mutation_rate, crossover_rate) -> None:
    f_opt = sys.float_info.min
    x_opt = None
    
    parent = []
    parent_f = []
    for i in range(population_size):

        # Initialization
        parent.append(np.random.randint(2, size = problem.meta_data.n_variables))
        parent_f.append(problem(parent[i]))


    while (problem.state.evaluations < budget):
        #selection(propotional selection)
        # offspring = None
        # f_min = min(parent_f)
        # f_sum = sum(parent_f) - (f_min - 0.001) * len(parent_f)
    
        # rw = [(parent_f[0] - f_min + 0.001)/f_sum]
        # for i in range(1,len(parent_f)):
        #     rw.append(rw[i-1] + (parent_f[i] - f_min + 0.001) / f_sum)
    
        # select_parent = []
        # for i in range(len(parent)) :
        #     r = np.random.uniform(0,1)
        #     index = 0
        #     # print(rw,r)
        #     while(r > rw[index]) :
        #         index = index + 1
        
        #     select_parent.append(parent[index].copy())
        
        # offspring = select_parent.copy()

        # tournament selection
        offspring = None
        select_parent = []
        k = 3  
        for i in range(len(parent)):
            if len(parent_f) < k:
                pre_select = np.random.choice(len(parent_f), len(parent_f), replace=False)
            else:
                pre_select = np.random.choice(len(parent_f), k, replace=False)

            max_f = sys.float_info.min
            index = pre_select[0] 
            for p in pre_select:
                if parent_f[p] > max_f:
                    index = p
                    max_f = parent_f[p]
            select_parent.append(parent[index].copy())

        offspring = select_parent.copy()


        #uniform crossover
        # for i in range(0,population_size - (population_size%2),2) :
        #     if(np.random.uniform(0,1) < crossover_rate):
        #         for j in range(len(offspring[i])): 
        #             if np.random.uniform(0,1) < 0.5:
        #                 t = offspring[i][j]
        #                 offspring[i][j] = offspring[i+1][j]
        #                 offspring[i+1][j] = t
        
        # # 1-point crossover
        for i in range(0, population_size - (population_size % 2), 2):
            if np.random.uniform(0, 1) < crossover_rate:
                point = np.random.randint(1, len(offspring[i]))  
                for j in range(point, len(offspring[i])):  
                    t = offspring[i][j]
                    offspring[i][j] = offspring[i+1][j]
                    offspring[i+1][j] = t

        # # Standard bit mutation using mutation rate p
        for i in range(population_size) :       
            for j in range(len(offspring[i])): 
                if np.random.uniform(0,1) < mutation_rate:
                    offspring[i][j] = 1 - offspring[i][j]

        parent = offspring.copy()
        for i in range(population_size):
            parent_f[i] = problem(parent[i])
            if parent_f[i] > f_opt:
                    f_opt = parent_f[i]
                    x_opt = parent[i].copy()
        
    print(f_opt,x_opt)
    return f_opt, x_opt

def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)
    l = logger.Analyzer(
        root="data",
        folder_name="run",
        algorithm_name="tourn_1point_stand_5_0.01_0.1",
        algorithm_info="Practical assignment of the EA course"
    )
    problem.attach_logger(l)
    return problem, l

if __name__ == "__main__":
    pop_size = 5
    mutation_rate = 0.01
    crossover_rate = 0.1

    F18, logger_F18 = create_problem(dimension=50, fid=18)
    for run in range(20):
        s_GA(F18, pop_size, mutation_rate, crossover_rate)
        F18.reset()
    logger_F18.close()

    F23, logger_F23 = create_problem(dimension=49, fid=23)
    for run in range(20):
        s_GA(F23, pop_size, mutation_rate, crossover_rate)
        F23.reset()
    logger_F23.close()
