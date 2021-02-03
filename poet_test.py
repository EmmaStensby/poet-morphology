from individuals.neural_network import Individual
from evaluators.bipedal_walker_fitness_behaviour_evaluator import Evaluator
from optimisers.ga import GAOptimiser
from algorithms.poet import POET
import numpy as np

if __name__ == "__main__":
    evaluator = Evaluator()
    seed = np.random.SeedSequence()
    algorithm = POET(GAOptimiser, (Individual, evaluator), seed)
    algorithm.run(100)
