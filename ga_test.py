from individuals.neural_network import Individual
from evaluators.bipedal_walker_fitness_behaviour_evaluator import Evaluator
from optimisers.ga import GAOptimiser
from box2D.bipedal_walker_hardcore_custom import BipedalWalkerCustom, Env_config
import numpy as np

if __name__ == "__main__":
    evaluator = Evaluator()
    seed = np.random.SeedSequence()
    env_config = Env_config(name='flat_ground', ground_roughness=0,
                            pit_gap=[], stump_width=[],
                            stump_height=[], stump_float=[],
                            stair_height=[], stair_width=[],
                            stair_steps=[],
                            leg_l_w=0, leg_l_h=0, leg_r_w=0, leg_r_h=0,
                            lower_l_w=0, lower_l_h=0, lower_r_w=0, lower_r_h=0)
    optimiser = GAOptimiser(Individual, env_config, evaluator, seed)
    optimiser.run(1000, video=True)
