import numpy as np
import utilities.utilities as ut

class Individual:
    def __init__(self, seed):
        self.ss = seed
        self.controller = NeuralNetwork(self.ss.spawn(1)[0])
        self.morphology = Morphology(self.ss.spawn(1)[0])
        
    def forward(self, inp, time=None):
        return self.controller.forward(inp)

    def mutate(self):
        self.controller.mutate()
        self.morphology.mutate()

    def recombine(self, other):
        nn1, nn2 = self.controller.recombine(other.controller)
        m1, m2 = self.morphology.recombine(other.morphology)
        c1 = Individual(self.ss.spawn(1)[0])
        c2 = Individual(self.ss.spawn(1)[0])
        c1.controller = nn1
        c2.controller = nn2
        c1.morphology = m1
        c2.morphology = m2
        return c1, c2
        
    def get_morphology(self):
        return self.morphology.morphology

    def compare_morphology(self, other):
        return self.morphology.compare(other.morphology)

class Morphology:
    def __init__(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.parameters = 8
        
        self.replace_rate = 0.0075
        self.scale_rate = 0.075
        self.mutate_power = 0.16

        self.genome = None
        self.morphology = None

        self._init_random()

    def _init_random(self):
        self.genome = self.rng.random(self.parameters)
        self.morphology = [0 for x in range(self.parameters)]
        self._unpack_genome()

    def _unpack_genome(self):
        for i in range(len(self.genome)):
            if i == 1 or i == 3 or i == 5 or i == 7:
                self.morphology[i] = (self.genome[i]*51) + 8.5
            if i == 0 or i == 4:
                self.morphology[i] = (self.genome[i]*12) + 2.0
            if i == 2 or i == 6:
                self.morphology[i] = (self.genome[i]*9.6) + 1.6
        
    def compare(self, other):
        difference = 0
        for n1, n2 in zip(self.morphology, other.morphology):
            d = n1 - n2
            if d < 0:
                d = -d
            difference += d
        return difference

    def mutate(self):
        self._replace()
        self._scale()
        self._unpack_genome()

    def _replace(self):
        for i in range(len(self.genome)):
            chance = self.rng.random()
            if chance < self.replace_rate:
                self.genome[i] = self.rng.random()
    
    def _scale(self):
        for i in range(len(self.genome)):
            chance = self.rng.random()
            if chance < self.scale_rate:
                self.genome[i] += (self.rng.random()*2*self.mutate_power)-self.mutate_power
                if self.genome[i] > 1.0:
                    self.genome[i] = 1.0 
                elif self.genome[i] < 0:
                    self.genome[i] = 0

    def recombine(self, other):
        c1 = Morphology(self.seed.spawn(1)[0])
        c2 = Morphology(self.seed.spawn(1)[0])
        p1, p2 = ut.uniform_crossover(self.genome, other.genome, self.seed.spawn(1)[0])
        c1.set_parameters(p1)
        c2.set_parameters(p2)
        return c1, c2

    def set_parameters(self, params):
        self.genome = params
        self._unpack_genome()

class NeuralNetwork:
    def __init__(self, seed, nodes=[24,40,40,4], activation=ut.identity, hyperparameters=[0.0075, 0.075, 0.2, 30.0, 1.0]):
        self.ss = seed
        self.rng = np.random.default_rng(self.ss)
        
        # Hyperparameters
        self.replace_rate = hyperparameters[0]
        self.scale_rate = hyperparameters[1]
        self.mutate_power = hyperparameters[2]
        self.max_weight_size = hyperparameters[3]
        self.init_weight_size = hyperparameters[4]
        
        # Data
        self.nodes = nodes
        self.weights = []
        self.biases = []

        self.genome = None

        # Count parameters
        self.total_weights = 0
        self.total_biases = 0
        for i in range(len(self.nodes)-1):
            self.total_weights += self.nodes[i]*self.nodes[i+1]
            self.total_biases += self.nodes[i+1]
        self.parameters =  self.total_weights + self.total_biases

        # Activation function
        self.activation = activation

        # Initialisation
        self._init_parameters()

    def _init_parameters(self):
        self.genome = (self.rng.random(self.parameters)*2*self.init_weight_size)-self.init_weight_size
        self._unpack_genome()
        
    def forward(self, inp):
        h_temp = inp
        for i in range(len(self.nodes)-1):
            h_temp = np.dot(self.weights[i], h_temp) + self.biases[i]
        return self.activation(h_temp)

    def _unpack_genome(self):
        self.weights = []
        self.biases = []
        cutoff = 0
        for i in range(len(self.nodes)-1):
            self.weights.append(list(np.asarray(self.genome[cutoff:cutoff+self.nodes[i]*self.nodes[i+1]]).reshape(self.nodes[i+1], self.nodes[i])))
            cutoff += self.nodes[i]*self.nodes[i+1]
            self.biases = list(np.asarray(self.genome[cutoff:cutoff+self.nodes[i+1]]).reshape(self.nodes[i+1]))
            cutoff += self.nodes[i+1]

    def set_parameters(self, params):
        self.genome = params
        self._unpack_genome()

    def mutate(self):
        self._replace()
        self._scale()
        self._unpack_genome()

    def _replace(self):
        for i in range(len(self.genome)):
            chance = self.rng.random()
            if chance < self.replace_rate:
                self.genome[i] = (self.rng.random()*2*self.init_weight_size)-self.init_weight_size
    
    def _scale(self):
        for i in range(len(self.genome)):
            chance = self.rng.random()
            if chance < self.scale_rate:
                self.genome[i] += (self.rng.random()*2*self.mutate_power)-self.mutate_power
                if self.genome[i] > self.max_weight_size:
                    self.genome[i] = self.max_weight_size
                elif self.genome[i] < -self.max_weight_size:
                    self.genome[i] = -self.max_weight_size
                    

    def recombine(self, nn):
        c1 = NeuralNetwork(self.ss.spawn(1)[0])
        c2 = NeuralNetwork(self.ss.spawn(1)[0])
        p1, p2 = ut.uniform_crossover(self.genome, nn.genome, self.ss.spawn(1)[0])
        c1.set_parameters(p1)
        c2.set_parameters(p2)
        return c1, c2
