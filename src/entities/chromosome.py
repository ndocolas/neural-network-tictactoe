import numpy as np

class Chromosome:
    """
    Envolve um vetor de weights_vector e seu fitness.
    Responsabilidade única: armazenar weights_vector e fitness.
    """
    def __init__(self, weights_vector: np.ndarray):
        self.weights_vector: np.ndarray = weights_vector
        self.fitness: float = 0.0

    def set_fitness(self, fitness: float):
        """Define o valor de fitness."""
        self.fitness = fitness
