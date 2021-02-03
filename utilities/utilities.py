import numpy as np

def sigmoid(v):
    return 1.0/(1.0 + (np.e**(-v)))

def identity(v):
    return v

def cap(v):
    c = np.pi
    new_v = []
    for val in v:
        if val > c:
            new_v.append(c)
        elif val < -c:
            new_v.append(-c)
        else:
            new_v.append(val)
    return new_v

def sin(v):
    return np.sin(v)

def one_point_crossover(v1, v2, seed):
    return _n_point_crossover(v1, v2, 1, seed)
    
def n_point_crossover(v1, v2, n, seed):
    rng = np.random.default_rng(seed)
    points = rng.choice(range(len(v1)), n, replace=False)
    switch = True
    n1 = []
    n2 = []
    for i in range(len(v1)):
        if i in points:
            switch = not switch
        if switch:
            n1.append(v1[i])
            n2.append(v2[i])
        else:
            n1.append(v2[i])
            n2.append(v1[i])
    if rng.random() > 0.5:
        return n1, n2
    return n2, n1

def uniform_crossover(v1, v2, seed):
    rng = np.random.default_rng(seed)
    n1 = []
    n2 = []
    for val1, val2 in zip(v2, v2):
        if rng.random() > 0.5:
            n1.append(val1)
            n2.append(val2)
        else:
            n1.append(val2)
            n2.append(val1)
    if rng.random() > 0.5:
        return n1, n2
    return n2, n1
