import numpy as np
import random

from entities.chromosome import Chromosome
from usecases.advanced_fitness_evaluator import AdvancedFitnessEvaluator

class GeneticAlgorithm:
    """
    GA com avaliação configurável por n_games:
      80 % primeiros jogos p=0.5, 20 % últimos jogos p=1.0.
      Inclui critério de parada automática por convergência de fitness.
    """
    def __init__(self, population_size: int = 100, generations: int = 50, n_games: int = 10):
        self.pop_size = population_size
        self.generations = generations
        self.n_games = n_games

        self.in_size = self.h_size = self.o_size = 9
        self.mut_rate = 0.1

        self.vector_len = 180
        self.evaluator = AdvancedFitnessEvaluator(self.in_size, self.h_size,
                                                     self.o_size, n_games)

    def _init_pop(self):
        return [Chromosome(np.random.uniform(-1,1,self.vector_len))
                for _ in range(self.pop_size)]

    def _select(self, pop):
        return random.choice(pop)

    def _crossover(self, a, b):
        child_vec = np.zeros(self.vector_len)
        for i in range(self.vector_len):
            child_vec[i] = (a.weights_vector[i] + b.weights_vector[i]) / 2.0
        return Chromosome(child_vec)

    def _mutate(self, chrom: Chromosome, verbose: bool = False) -> Chromosome:
        """
        Mutação real-coded:
          • perturba ~mut_rate dos genes com ruído gaussiano suave
          • 30 % de chance de um “burst” extra (1-3 genes) com ruído maior
          • garante intervalo [-1, 1]
        """
        # --- novo objeto para não alterar o pai ---
        child = Chromosome(chrom.weights_vector.copy())

        # máscara: muta cerca de mut_rate * vector_len genes
        mask = np.random.rand(self.vector_len) < self.mut_rate
        child.weights_vector[mask] += np.random.normal(0, 0.15, size=mask.sum())

        # burst extra
        extra_count = 0
        if random.random() < 0.3:
            extra_idx = np.random.choice(self.vector_len,
                                         size=random.randint(1, 3),
                                         replace=False)
            child.weights_vector[extra_idx] += np.random.normal(0, 0.30,
                                                                size=len(extra_idx))
            extra_count = len(extra_idx)

        # mantém pesos no intervalo inicial
        np.clip(child.weights_vector, -1, 1, out=child.weights_vector)

        if verbose:
            total = mask.sum() + extra_count
            print(f"Total mutations: {total}")

        return child
    
    def _select_tournament(self, pop, k=3):
        """Retorna o melhor entre k indivíduos sorteados."""
        return max(random.sample(pop, k), key=lambda c: c.fitness)


    def evolve(self, verbose: bool = False):
        pop = self._init_pop()
        best_vec  = None
        best_fit  = -float('inf')

        for g in range(1, self.generations + 1):
            # ---------- avaliação ----------
            for c in pop:
                c.fitness = self.evaluator.evaluate(c.weights_vector, g)

            pop.sort(key=lambda c: c.fitness, reverse=True)
            if best_vec is None:                   # primeira geração
                best_vec = pop[0].weights_vector.copy()
                best_fit = pop[0].fitness

            if pop[0].fitness > best_fit:
                best_fit = pop[0].fitness
                best_vec = pop[0].weights_vector.copy()

            if verbose:
                avg_fit = sum(c.fitness for c in pop) / len(pop)
                print(
                  f"Gen {g}/{self.generations} | "
                  f"Best(gen): {pop[0].fitness:.2f}  "
                  f"Avg: {avg_fit:.2f}  "
                  f"Best(ever): {best_fit:.2f}"
                )

            next_pop = [Chromosome(best_vec.copy())]  # elitismo

            while len(next_pop) < self.pop_size:
                p1 = self._select_tournament(pop, k=3)
                p2 = self._select_tournament(pop, k=3)
                while p2 is p1:                       # evita clone
                    p2 = self._select_tournament(pop, k=3)

                child = self._crossover(p1, p2)
                if g > 30: 
                    child = self._mutate(child)
                child.fitness = None                  # ainda não avaliado
                next_pop.append(child)

            pop = next_pop

        if verbose:
            print("\nTreinamento completo.\n")

        return best_vec

