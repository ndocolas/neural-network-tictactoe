import random
import numpy as np

from entities.chromosome import Chromosome
from usecases.advanced_fitness_evaluator import AdvancedFitnessEvaluator
from adapters.minimax_trainer import MinimaxTrainer

class GeneticAlgorithm:
    """
    GA para Tic-Tac-Toe usando AdvancedFitnessEvaluator e hall-of-fame.
    """
    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 9,
        output_size: int = 9,
        population_size: int = 50,
        generations: int = 100,
        tournament_size: int = 3,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.5,
        n_games: int = 5
    ):
        self.in_size      = input_size
        self.h_size       = hidden_size
        self.o_size       = output_size
        self.pop_size     = population_size
        self.generations  = generations
        self.tourn_size   = tournament_size
        self.cx_rate      = crossover_rate
        self.mut_rate     = mutation_rate
        self.mut_scale    = mutation_scale
        self.n_games      = n_games

        # comprimento do genome
        self.genome_len = hidden_size * (input_size + 1) \
                        + output_size * (hidden_size + 1)

        # nosso avaliador avançado
        self.evaluator = AdvancedFitnessEvaluator(
            input_size, hidden_size, output_size, n_games
        )

    def _init_population(self) -> list[Chromosome]:
        return [
            Chromosome(np.random.uniform(-1, 1, self.genome_len))
            for _ in range(self.pop_size)
        ]

    def _select_parent(self, pop: list[Chromosome]) -> Chromosome:
        contenders = random.sample(pop, self.tourn_size)
        return max(contenders, key=lambda c: c.fitness)

    def _crossover(self, p1: Chromosome, p2: Chromosome) -> Chromosome:
        mask  = np.random.rand(self.genome_len) < self.cx_rate
        alpha = np.random.rand(self.genome_len)
        child = np.where(
            mask,
            alpha * p1.genome + (1 - alpha) * p2.genome,
            p1.genome.copy()
        )
        return Chromosome(child)

    def _mutate(self, chrom: Chromosome):
        for i in range(self.genome_len):
            if random.random() < self.mut_rate:
                chrom.genome[i] += np.random.normal(scale=self.mut_scale)

    def evolve(self, verbose: bool = False) -> np.ndarray:
        pop = self._init_population()

        best_ever_genome  = None
        best_ever_fitness = -float('inf')
        half = self.generations // 2

        for gen in range(1, self.generations + 1):
            p_use = 1.0 if gen <= half else 0.5

            # avalia toda a população
            for chrom in pop:
                fitness = self.evaluator.evaluate(chrom, p_use)
                chrom.set_fitness(fitness)

            # ordena por fitness e atualiza hall‐of‐fame
            pop.sort(key=lambda c: c.fitness, reverse=True)
            if pop[0].fitness > best_ever_fitness:
                best_ever_fitness = pop[0].fitness
                best_ever_genome  = pop[0].genome.copy()

            if verbose:
                print(f"Gen {gen}/{self.generations} | "
                      f"p_minimax={p_use:.2f} | "
                      f"Best gen: {pop[0].fitness:.3f} | "
                      f"Best ever: {best_ever_fitness:.3f}")

            # monta próxima geração com elitismo forte
            next_pop = []

            # injeta o campeão absoluto
            champ = Chromosome(best_ever_genome.copy())
            champ.set_fitness(best_ever_fitness)
            next_pop.append(champ)

            # preenche via torneio, crossover e mutação
            while len(next_pop) < self.pop_size:
                p1    = self._select_parent(pop)
                p2    = self._select_parent(pop)
                child = self._crossover(p1, p2)
                self._mutate(child)
                next_pop.append(child)

            pop = next_pop

        if verbose:
            print("\nTreinamento completo.\n")

        return best_ever_genome
