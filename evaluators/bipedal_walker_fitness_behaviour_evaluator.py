import gym
import numpy as np
gym.logger.set_level(40)
from box2D.bipedal_walker_hardcore_custom import BipedalWalkerCustom, Env_config

class Evaluator:
    def __init__(self):
        self.max_steps = 1000
        self.stability_of_evaluation = 4
        
    def evaluate(self, individual, env_config, seed):
        env_config = self.apply_morphology_to_env_config(env_config, individual.get_morphology())
        env = BipedalWalkerCustom(env_config)

        rng = np.random.default_rng(seed)
        ii32 = np.iinfo(np.int32)

        env_seed = int(rng.integers(ii32.max))
        env.seed(env_seed)
        
        fitness_for_runs = []

        # Do the simulations
        for run in range(self.stability_of_evaluation):
            state = env.reset()
            done = False
            total_reward = 0
            step = 0
            while not done and step <= self.max_steps:
                action = individual.forward(state, step)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                total_reward += reward
                step += 1
            fitness_for_runs.append(total_reward)
            
        mean_reward = sum(fitness_for_runs)/self.stability_of_evaluation
        return (mean_reward, self.stability_of_evaluation)

    def video(self, individual, env_config, seed):
        env_config = self.apply_morphology_to_env_config(env_config, individual.get_morphology())
        env = BipedalWalkerCustom(env_config)

        rng = np.random.default_rng(seed)
        ii32 = np.iinfo(np.int32)

        env_seed = int(rng.integers(ii32.max))
        env.seed(env_seed)

        # Do the simulations
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        while not done and step <= self.max_steps:
            env.render()
            action = individual.forward(state, step)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            step += 1
    
        env.render(close=True)

        print("Video fitness: ", total_reward)

    def apply_morphology_to_env_config(self, env_config, morphology):
        env_config = Env_config(name=env_config.name, ground_roughness=env_config.ground_roughness,
                                pit_gap=env_config.pit_gap, stump_width=env_config.stump_width,
                                stump_height=env_config.stump_height, stump_float=env_config.stump_float,
                                stair_height=env_config.stair_height, stair_width=env_config.stair_width,
                                stair_steps=env_config.stair_steps,
                                leg_l_w=morphology[0], leg_l_h=morphology[1], leg_r_w=morphology[2], leg_r_h=morphology[3],
                                lower_l_w=morphology[4], lower_l_h=morphology[5], lower_r_w=morphology[6], lower_r_h=morphology[7])
        return env_config
