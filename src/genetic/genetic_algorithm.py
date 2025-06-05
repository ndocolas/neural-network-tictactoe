import random
from typing import List
from utils.fitness import evaluate_population
from neural_network.network import NeuralNetwork

POP_SIZE = 50
CHROMO_SIZE = 180

def initialize_population(pop_size: int = POP_SIZE) -> List[List[float]]:
    return [[random.uniform(-1, 1) for _ in range(CHROMO_SIZE)] for _ in range(pop_size)]

def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    return [(a + b) / 2 for a, b in zip(parent1, parent2)]

def mutate(chromosome: List[float], rate: float = 0.05) -> List[float]:
    return [gene + random.uniform(-0.1, 0.1) if random.random() < rate else gene for gene in chromosome]

def select_parents(population: List[List[float]], fitnesses: List[float]) -> List[List[float]]:
    selected = random.choices(population, weights=fitnesses, k=2)
    return selected

def run_evolution(generations: int = 100, population_size: int = POP_SIZE) -> List[float]:
    population = initialize_population(population_size)

    for generation in range(generations):
        fitnesses = evaluate_population(population)
        new_population = []

        elite = population[fitnesses.index(max(fitnesses))]
        new_population.append(elite)

        while len(new_population) < population_size:
            p1, p2 = select_parents(population, fitnesses)
            child = mutate(crossover(p1, p2))
            new_population.append(child)

        population = new_population

    final_fitnesses = evaluate_population(population)
    best_index = final_fitnesses.index(max(final_fitnesses))
    return population[best_index]
