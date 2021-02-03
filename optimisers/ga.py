import numpy as np
import multiprocessing as mp
from multiprocessing import pool

class NoDaemonProcess(mp.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class Pool(pool.Pool):
    Process = NoDaemonProcess

class GAOptimiser:
    def __init__(self, controller_type, evaluator, seed):
        self.env_config = None
        
        self.controller_type = controller_type

        self.ss = seed
        self.main_seed = self.ss.spawn(1)[0]
        self.rng = np.random.default_rng(self.main_seed)
        
        self.evaluator = evaluator

        self.threads = mp.cpu_count()

        self.population_size = 192

        seeds = self.ss.spawn(self.population_size)
        self.population = [controller_type(seeds[i]) for i in range(self.population_size)]

        self.precalculated_fitness = None

        self.total_generation = 0

        self.evaluations_spent = 0

    def seed(self, seed):
        self.ss = seed
        self.main_seed = self.ss.spawn(1)[0]
        self.rng = np.random.default_rng(self.main_seed)

    def set_env_config(self, env_config):
        self.env_config = env_config

    def run(self, generations, current_evaluations, verbose=True, video=False):
        self.evaluations_spent = current_evaluations
        self.precalculated_fitness = None
        for i in range(generations):
            self.total_generation += 1
            # Evaluate Population
            fitness, _ = self.evaluate_parallel(self.population)
            # Select Parents
            parents = self.select_parents_tournament(self.population, fitness)
            # Create Children
            children = self.create_children(parents)
            # Evaluate Children
            child_fitness, _ = self.evaluate_parallel(children)
            # Select Children
            self.population = self.select_children_crowding(children, child_fitness, self.population, fitness, save_precalculated=True)
            # Print status
            if verbose:
                print("Env: {}, Evaluations: {}, Max fitness: {}, Mean fitness: {}".format(self.env_config.name, self.evaluations_spent, max(self.precalculated_fitness), np.mean(self.precalculated_fitness)))
            if video and i%10 == 0:
                index = self.precalculated_fitness.index(max(self.precalculated_fitness))
                self.evaluator.video(self.population[index], self.env_config, self.ss.spawn(1)[0])
        return (self.evaluations_spent - current_evaluations)

    def evaluate_parallel(self, population):
        parameters = []
        seeds = self.ss.spawn(len(population))
        for i in range(len(population)):
            parameters.append((population[i], self.env_config, seeds[i]))
        with Pool(self.threads) as p:
            fitness_eval_list = p.starmap(self.evaluator.evaluate, parameters)
        fitness = []
        evaluations = 0
        for pair in fitness_eval_list:
            fitness.append(pair[0])
            evaluations += pair[1]
        self.evaluations_spent += evaluations
        return fitness, evaluations

    def select_parents_tournament(self, population, fitness, tournament_size=5):
        parents = []
        for i in range(self.population_size):
            tournament = self.rng.choice(range(self.population_size), tournament_size, replace=False)
            best = None
            best_fitness = None
            for j in tournament:
                if best is None:
                    best = population[j]
                    best_fitness = fitness[j]
                else:
                    if fitness[j] > best_fitness:
                         best = population[j]
                         best_fitness = fitness[j]
            parents.append(best)       
        return parents

    def create_children(self, parents):
        self.rng.shuffle(parents)
        children = []
        for i in range(int(len(parents)/2)):
            c1, c2 = parents[i*2].recombine(parents[i*2+1])
            c1.mutate()
            c2.mutate()
            children.append(c1)
            children.append(c2)
        return children

    def select_children_crowding(self, children, children_fitness, population, population_fitness, save_precalculated=True, w=20):
        new_population = population
        new_fitness = population_fitness
        for child, child_fitness in zip(children, children_fitness):
            tournament = self.rng.choice(range(len(new_population)), w, replace=False)
            lowest_diff = None
            lowest = None
            for j in tournament:
                if lowest is None:
                    lowest = j
                    lowest_diff = child.compare_morphology(new_population[j])
                else:
                    diff = child.compare_morphology(new_population[j])
                    if lowest_diff > diff:
                        lowest_diff = diff
                        lowest = j
            if child_fitness > new_fitness[lowest]:
                new_fitness[lowest] = child_fitness
                new_population[lowest] = child
            
        if save_precalculated:
            self.precalculated_fitness = new_fitness
        return new_population
