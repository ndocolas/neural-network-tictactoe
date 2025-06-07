import numpy as np

class Chromosome:
    """
    Envolve um vetor de genome e seu fitness.
    Responsabilidade única: armazenar genome e fitness.
    """
    def __init__(self, genome: np.ndarray):
        self.genome: np.ndarray = genome
        self.fitness: float = 0.0

    def set_fitness(self, fitness: float):
        """Define o valor de fitness."""
        self.fitness = fitness
