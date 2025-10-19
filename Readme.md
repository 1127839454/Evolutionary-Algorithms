# Submitted Code/File Description


## Submitted Files
The submission includes the following files:

### `GA.py`
Implements the Genetic Algorithm to solve the LABS and N-Queens problems.Includes methods for selection, crossover, and mutation.
- **Selection Methods**: 
    - Proportional Selection
    - Tournament Selection 
- **Crossover Methods**: 
    - Uniform Crossover
    - 1-Point Crossover 
- **Mutation**: 
    - Standard Bit Mutation
- **Hyperparameters**:
    - Population size
    - Mutation rate
    - Crossover rate


### `tuning.py`
Automates hyperparameter tuning for GA.Explores combinations of population size, mutation rate, and crossover rate.

### `ES.py`
Implements the Evolution Strategy. Configured to solve the Katsuura problem.
- **Recombination Methods**:
  - Intermediate
  - Global Intermediate
  - Discrete
  - Global Discrete 
- **Mutation Methods**:
  - One Sigma Mutation
  - Individual Sigma Mutation 
- **Selection Strategies**:
  - \((\mu+\lambda)\) 
  - \((\mu, \lambda)\)
- **Hyperparameters**:
  - Parent size (\(\mu\))
  - Offspring size (\(\lambda\))
  - Learning rates (\(\tau_p\) and \(\tau\))

## Running the Code
To run `GA.py`, first you need to determain the configuration conbinations you want. In `GA.py`, we implemented multiple methods for selection, crossover, and mutation, but we can only use one for each of them. So you need to select the configuration you want to use, uncomment it, and comment out the methods that are not needed. For example, you want to use `Proportional Selection + Uniform Crossover`, you must comment out `Tournament Selection`, `1-Point Crossover`. Then you can use `python GA.py` command to run it.

`tuning.py` aims to get the parameter pair that can maximum the average best fitness of `GA.py`. So you can use `python tuning.py` to run it, and use the parameters you get to fill them in `GA.py`, and then run `GA.py` to get the result. The result of `tuning.py` include the average best fitness of F18 and F23, the best fitness of F18 and the best fitness of F23. The last two result is not mandatory, we keep it to see what's the best fitness we can get.

Similar to running `GA.py`, to run `ES.py`, you also need to comment out the methods you don't need. It should be noticed that if you want to use `individual sigma mutation`, I wrote special versions of `initialization, discrete recombination, global discrete recombination, tau, tau_p` for the `individual sigma mutation`. So you have to use them when you want to use it. `intermediate recombination` and `global intermediate recombination` can be used for both `one-sigma mutation` and `individual sigma mutation`.