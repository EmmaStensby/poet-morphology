import numpy as np
from copy import deepcopy
import pickle
from box2D.bipedal_walker_hardcore_custom import Env_config

class Pair:
    """ A POET pair consisting of an environment and an agent. """
    def __init__(self, seed):
        self.environment = None
        self.agent = None
        self.fitness = None
        self.seed = seed

    def init_first(self, agent_type, agent_parameters):
        self.environment = Env_config(name=str(1), ground_roughness=0, pit_gap=[], stump_width=[], stump_height=[],
                                      stump_float=[], stair_height=[], stair_width=[], stair_steps=[], leg_l_w=0,
                                      leg_l_h=0, leg_r_w=0, leg_r_h=0, lower_l_w=0,
                                      lower_l_h=0, lower_r_w=0, lower_r_h=0)
        self.agent = agent_type(*agent_parameters, self.seed)

class POET:
    def __init__(self, agent_type, agent_parameters, seed):
        self.seed = seed
        self.main_seed = self.seed.spawn(1)[0]
        self.rng = np.random.default_rng(self.main_seed)
        
        # Parameters
        self.mutation_chance = 0.2
        self.transfer_frequency = 5
        self.create_frequency = 40
        self.reproduction_criterion = 200
        self.difficulty_criterion_low = 50
        self.difficulty_criterion_high = 300
        self.num_create_environments = 20
        self.num_children_add = 2
        self.max_pair_population_size = 20
        self.k = 5

        self.agent_type = agent_type
        
        # The pairs of environments and agents
        self.pairs = []
        first_pair = Pair(self.seed.spawn(1)[0])
        first_pair.init_first(self.agent_type, agent_parameters)
        self.pairs.append(first_pair)

        # The archive with all environments that have ever existed in the pair population
        self.environment_archive = []
        self.environment_archive.append(first_pair.environment)

        self.total_environments_created = 1

        self.total_generation = 0

        self.total_evaluations = 0

    def run(self, generations):
        assert(self.create_frequency%self.transfer_frequency == 0)
        assert(generations%self.transfer_frequency == 0)
        for i in range(int(generations/self.transfer_frequency)):
            # Transfer
            if self.total_generation%self.transfer_frequency == 0:
                self.transfer()
            # Create new environments
            if self.total_generation%self.create_frequency == 0:
                self.create_environments()
            # Train
            self.train_agents(self.transfer_frequency)
            # Create checkpoint
            if self.total_generation%self.transfer_frequency == 0:
                self.save_checkpoint(self.total_generation)
            self.total_generation += self.transfer_frequency

    def save_checkpoint(self, gen):
        path = "checkpoints/cp_gen_{}.pkl".format(gen)
        f = open(path, "wb")
        pickle.dump(self, f)
        f.close()

    def create_environments(self):
        # Find eligible pairs
        eligible_pairs = []
        for pair in self.pairs:
            if (pair.fitness is not None) and (pair.fitness > self.reproduction_criterion):
                eligible_pairs.append(pair)
        # Create child environments
        child_environments = []
        if len(eligible_pairs) > 0:
            selected_pairs = np.random.choice(eligible_pairs, self.num_create_environments, replace=True)
            for pair in selected_pairs:
                child_environments.append(self.mutate(pair.environment))
        # Find agents for the children and test them against the minimal criteria
        eligible_child_pairs = []
        for environment in child_environments:
            child_pair = Pair(self.seed.spawn(1)[0])
            child_pair.environment = environment
            best_agent = None
            best_fitness = None
            for pair in self.pairs:
                child_pair.agent = deepcopy(pair.agent)
                child_pair.agent.seed(child_pair.seed)
                fitness = self.evaluate_pair(child_pair)
                if (best_fitness is None) or (fitness > best_fitness):
                    best_agent = child_pair.agent
                    best_fitness = fitness
            if (best_fitness > self.difficulty_criterion_low) and (best_fitness < self.difficulty_criterion_high):
                child_pair.agent = best_agent
                eligible_child_pairs.append(child_pair)
        # Select child environments to add to pair population
        sorted_child_pairs = self.sort_child_pairs(eligible_child_pairs)
        added = 0
        for child in sorted_child_pairs:
            if added < self.num_children_add:
                self.pairs.append(child)
                self.environment_archive.append(child.environment)
                if len(self.pairs) > self.max_pair_population_size:
                    self.pairs.pop(0)
            added += 1

    # FIX mutate
    def mutate(self, env):
        mutation_chance = self.mutation_chance

        new_gr = env.ground_roughness
        pit_gap = env.pit_gap
        stump_width = env.stump_width
        stump_height = env.stump_height
        stair_width = env.stair_width
        stair_height = env.stair_height
        new_steps = env.stair_steps

        mut_ground = True
        mut_gap = True
        mut_stump = True
        mut_stair = True
        
        # Ground
        mutate_ground = np.random.rand()
        if mut_ground and mutate_ground < mutation_chance:
            min_ground_roughness = 0
            max_ground_roughness = 10
            new_gr = env.ground_roughness + ((np.random.rand()*1.2)-0.6)
            if new_gr < min_ground_roughness:
                new_gr = min_ground_roughness
            if new_gr > max_ground_roughness:
                new_gr = max_ground_roughness
        
        # Pit Gap
        mutate_pit = np.random.rand()
        if mut_gap and mutate_pit < mutation_chance:
            min_pit_0 = 0.1
            max_pit_0 = 10.0
            min_pit_1 = 0.8
            max_pit_1 = 10.0
            if len(env.pit_gap) == 0:
                pit_gap = [0.1, 0.8]
            else:
                new_pit_0 = env.pit_gap[0] + (random.choice([1,-1])*0.4)
                new_pit_1 = env.pit_gap[1] + (random.choice([1,-1])*0.4)
                if new_pit_0 < min_pit_0:
                    new_pit_0 = min_pit_0
                if new_pit_0 > max_pit_0:
                    new_pit_0 = max_pit_0
                if new_pit_1 < min_pit_1:
                    new_pit_1 = min_pit_1
                if new_pit_1 > max_pit_1:
                    new_pit_1 = max_pit_1
                pit_gap = [new_pit_0, new_pit_1]
            if pit_gap[0] > pit_gap[1]:
                st0 = pit_gap[0]
                pit_gap[0] = pit_gap[1]
                pit_gap[1] = st0
            
        # Stump Height
        mutate_stump_h = np.random.rand()
        if mut_stump and mutate_stump_h < mutation_chance:
            min_stump_h_0 = 0.1
            max_stump_h_0 = 5.0
            min_stump_h_1 = 0.4
            max_stump_h_1 = 5.0
            if len(env.stump_height) == 0:
                stump_width = [1, 2]
                stump_height = [0.1, 0.4]
            else:
                new_stump_h_0 = env.stump_height[0] + (random.choice([1,-1])*0.2)
                new_stump_h_1 = env.stump_height[1] + (random.choice([1,-1])*0.2)
                if new_stump_h_0 < min_stump_h_0:
                    new_stump_h_0 = min_stump_h_0
                if new_stump_h_0 > max_stump_h_0:
                    new_stump_h_0 = max_stump_h_0
                if new_stump_h_1 < min_stump_h_1:
                    new_stump_h_1 = min_stump_h_1
                if new_stump_h_1 > max_stump_h_1:
                    new_stump_h_1 = max_stump_h_1
                stump_height = [new_stump_h_0, new_stump_h_1]
            if stump_height[0] > stump_height[1]:
                st0 = stump_height[0]
                stump_height[0] = stump_height[1]
                stump_height[1] = st0
            
        # Stair Height
        mutate_stair_h = np.random.rand()
        if mut_stair and mutate_stair_h < mutation_chance:
            min_stair_h_0 = 0.1
            max_stair_h_0 = 5.0
            min_stair_h_1 = 0.4
            max_stair_h_1 = 5.0
            if len(env.stair_height) == 0:
                stair_width = [1, 2]
                stair_height = [0.1, 0.4]
                new_steps = [2]
            else:
                new_stair_h_0 = env.stair_height[0] + (random.choice([1,-1])*0.2)
                new_stair_h_1 = env.stair_height[1] + (random.choice([1,-1])*0.2)
                if new_stair_h_0 < min_stair_h_0:
                    new_stair_h_0 = min_stair_h_0
                if new_stair_h_0 > max_stair_h_0:
                    new_stair_h_0 = max_stair_h_0
                if new_stair_h_1 < min_stair_h_1:
                    new_stair_h_1 = min_stair_h_1
                if new_stair_h_1 > max_stair_h_1:
                    new_stair_h_1 = max_stair_h_1
                stair_height = [new_stair_h_0, new_stair_h_1]
            if stair_height[0] > stair_height[1]:
                st0 = stair_height[0]
                stair_height[0] = stair_height[1]
                stair_height[1] = st0

        # Steps
        mutate_steps = np.random.rand()
        if mut_stair and mutate_steps < mutation_chance:
            min_steps = 2
            max_steps = 9
            if len(env.stair_steps) == 0:
                new_steps = [2]
                stair_width = [1, 2]
                stair_height = [0.1, 0.4]
            else:
                new_steps = [env.stair_steps[0] + (random.choice([1,-1]))]
                if new_steps[0] < min_steps:
                    new_steps[0] = min_steps
                if new_steps[0] > max_steps:
                    new_steps[0] = max_steps

        self.total_environments_created += 1
        new_env = Env_config(name=str(self.total_environments_created), ground_roughness=new_gr, pit_gap=pit_gap, stump_width=stump_width,
                             stump_height=stump_height, stump_float=env.stump_float, stair_height=stair_height,
                             stair_width=stair_width, stair_steps=new_steps, leg_l_w=env.leg_l_w,
                             leg_l_h=env.leg_l_h, leg_r_w=env.leg_r_w, leg_r_h=env.leg_r_h, lower_l_w=env.lower_l_w,
                             lower_l_h=env.lower_l_h, lower_r_w=env.lower_r_w, lower_r_h=env.lower_r_h)

        return new_env
    
    def evaluate_pair(self, pair):
        pair.agent.set_env_config(pair.environment)
        fitness_list, evaluations = pair.agent.evaluate_parallel(pair.agent.population)
        self.total_evaluations += evaluations
        return max(fitness_list)
    
    def sort_child_pairs(self, pairs):
        # Remove already existing environments
        pruned_pairs = []
        for pair in pairs:
            if(not self.is_in_archive(pair.environment)):
                pruned_pairs.append(pair)
        # Compute novelty for the children
        novelties = []
        for pair in pruned_pairs:
            novelties.append(self.compute_novelty(pair.environment))
        # Sort children based on novelty
        sorted_pairs = []
        for i in range(len(novelties)):
            index = novelties.index(max(novelties))
            sorted_pairs.append(pruned_pairs.pop(index))
            novelties.pop(index)
        return sorted_pairs

    def is_in_archive(self, env):
        # Check if the environment already exists in the archive
        for environment in self.environment_archive:
            if self.compare_envs(environment, env) == 0:
                return True
        return False

    def compute_novelty(self, env):
        # Compute the novelty of an environment with regards to the archive
        # Novelty is the mean difference from the 5 nearest neighbours
        differences = []
        for environment in self.environment_archive:
            differences.append(self.compare_envs(environment, env))
        novelty = 0
        k = self.k
        if len(differences) < k:
            k = len(differences)
        for i in range(k):
            novelty_i = min(differences)
            differences.pop(differences.index(novelty_i))
            novelty += novelty_i/k
        return novelty

    def compare_envs(self, env1, env2):
        # Find the difference between two environments
        diff_num = 0
        diff_num += (env1.ground_roughness - env2.ground_roughness)**2
        diff_num += self.compare_feature_2(env1.pit_gap, env2.pit_gap)
        diff_num += self.compare_feature_2(env1.stump_height, env2.stump_height)
        diff_num += self.compare_feature_2(env1.stair_height, env2.stair_height)
        diff_num += self.compare_feature_1(env1.stair_steps, env2.stair_steps)
        return np.sqrt(diff_num)

    def compare_feature_1(self, f1, f2):
        diff_num = 0
        if (len(f1) == 1) and (len(f2) == 1):
            diff_num += (f1[0] - f2[0])**2
        elif (len(f1) == 2):
            diff_num += (f1[0])**2
        elif (len(f2) == 2):
            diff_num += (f2[0])**2
        return diff_num
    
    def compare_feature_2(self, f1, f2):
        diff_num = 0
        if (len(f1) == 2) and (len(f2) == 2):
            diff_num += (f1[0] - f2[0])**2
            diff_num += (f1[1] - f2[1])**2
        elif (len(f1) == 2):
            diff_num += (f1[0])**2
            diff_num += (f1[1])**2
        elif (len(f2) == 2):
            diff_num += (f2[0])**2
            diff_num += (f2[1])**2
        return diff_num
    
    def train_agents(self, generations):
        for pair in self.pairs:
            # Set environments
            pair.agent.set_env_config(pair.environment)
            # Train agents
            evaluations = pair.agent.run(generations, self.total_evaluations, verbose=True)
            self.total_evaluations += evaluations
            # Set fitness
            pair.fitness = max(pair.agent.precalculated_fitness)

    def transfer(self):
        # Direct transfer
        if len(self.pairs) > 1:
            for pair in self.pairs:
                best_agent = None
                best_fitness = None
                for transfer_pair in self.pairs:
                    temp_test_pair = Pair(self.seed.spawn(1)[0])
                    temp_test_pair.environment = pair.environment
                    temp_test_pair.agent = deepcopy(transfer_pair.agent)
                    temp_test_pair.agent.seed(temp_test_pair.seed)
                    fitness = self.evaluate_pair(temp_test_pair)
                    if (best_fitness is None) or (best_fitness < fitness):
                        best_agent = temp_test_pair.agent
                        best_fitness = fitness
                pair.agent = best_agent
