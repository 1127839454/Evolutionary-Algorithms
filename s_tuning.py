from typing import Tuple
import numpy as np
from s_GA import create_problem, s_GA

np.random.seed(42)
budget = 50000000

hyperparameter_space = {
    "population_size": [5, 10],
    "mutation_rate": [0.01, 0.1,  0.5],
    "crossover_rate": [0.1,0.65, 0.8, 0.9]
}

def tune_hyperparameters() -> Tuple[Tuple[float, Tuple[int, float, float]], Tuple[float, Tuple[int, float, float]], Tuple[float, Tuple[int, float, float]]]:
    best_avg_score = -float('inf')
    best_avg_params = None

    best_f18_score = -float('inf')
    best_f18_params = None

    best_f23_score = -float('inf')
    best_f23_params = None

    F18, logger_F18 = create_problem(dimension=50, fid=18)
    F23, logger_F23 = create_problem(dimension=49, fid=23)

    for pop_size in hyperparameter_space['population_size']:
        for mutation_rate in hyperparameter_space['mutation_rate']:
            for crossover_rate in hyperparameter_space['crossover_rate']:
                global budget
                if budget < 40 * 5000:
                    print("end tuning due to budget limit")
                    break

                scores_F18 = []
                for _ in range(20):
                    score, _ = s_GA(F18, pop_size, mutation_rate, crossover_rate)
                    scores_F18.append(score)
                    F18.reset()
                    budget -= 5000
                avg_score_F18 = sum(scores_F18) / len(scores_F18)

                if avg_score_F18 > best_f18_score:
                    print("true")
                    best_f18_score = avg_score_F18
                    best_f18_params = (pop_size, mutation_rate, crossover_rate)

                scores_F23 = []
                for _ in range(20):
                    score, _ = s_GA(F23, pop_size, mutation_rate, crossover_rate)
                    scores_F23.append(score)
                    F23.reset()
                    budget -= 5000
                avg_score_F23 = sum(scores_F23) / len(scores_F23)

                if avg_score_F23 > best_f23_score:
                    best_f23_score = avg_score_F23
                    best_f23_params = (pop_size, mutation_rate, crossover_rate)

                avg_score = (avg_score_F18 + avg_score_F23) / 2
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_avg_params = (pop_size, mutation_rate, crossover_rate)

    logger_F18.close()
    logger_F23.close()

    return (best_avg_score, best_avg_params), (best_f18_score, best_f18_params), (best_f23_score, best_f23_params)

if __name__ == "__main__":
    (best_avg, avg_params), (best_f18, f18_params), (best_f23, f23_params) = tune_hyperparameters()

    print(f"Best Average Score: {best_avg}, Parameters: {avg_params}")
    print(f"Best F18 Score: {best_f18}, Parameters: {f18_params}")
    print(f"Best F23 Score: {best_f23}, Parameters: {f23_params}")
