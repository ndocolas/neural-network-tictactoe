from usecases.score_evaluator import ScoreEvaluator
from entities.chromosome import Chromosome
from pathlib import Path
from typing import List
import numpy as np
import random
import os

class GeneticAlgorithm:
    """
    Algoritmo Genético:

      • Avaliação por n_games (80 % p=0.5, 20 % p=1.0)
      • Parada por número de gerações
      • Elitismo (melhor indivíduo segue intacto)
      • Uniform Crossover real-coded
      • Mutação Gaussiana + “burst” adicional
      • CSV por geração no formato IA,Score
    """

    def __init__(
        self,
        population_size: int,
        generations: int,
        n_games: int,
        out_dir: str | os.PathLike = "populations",
        seed: int | None = None,
    ):
        # ----- Hiperparâmetros principais -----
        self.pop_size = population_size
        self.generations = generations
        self.n_games = n_games

        # Arquitetura fixa 9-9-9 → 180 pesos
        self.in_size = self.h_size = self.o_size = 9
        self.vector_len = 180

        # Taxas do GA
        self.mut_rate = 0.10

        # Avaliador de fitness
        self.evaluator = ScoreEvaluator(
            self.in_size, self.h_size, self.o_size, n_games
        )

        # Pasta de saída
        self.out_path = Path(out_dir)
        self.out_path.mkdir(parents=True, exist_ok=True)

        # Semente opcional p/ reprodutibilidade
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _init_pop(self) -> List[Chromosome]:
        return [Chromosome(np.random.uniform(-1, 1, self.vector_len))
            for _ in range(self.pop_size)]

    def _select_tournament(self, pop: List[Chromosome], k: int = 2) -> Chromosome:
        """Torneio de tamanho *k* (default = 2)."""
        return max(random.sample(pop, k), key=lambda c: c.score)

    def _crossover(self, dad: Chromosome, mom: Chromosome):
        child_vec = np.zeros(self.vector_len)
        for i in range(self.vector_len):
            child_vec[i] = (dad.weights_vector[i] + mom.weights_vector[i]) / 2.0
        return Chromosome(child_vec)

    def _mutate(self, chrom: Chromosome, verbose: bool = False) -> None:
        """
        Mutação real-coded in-place:
          • Perturba ~mut_rate dos genes com N(0, 0.15)
          • 30 % de chance de “burst” (1-3 genes) com N(0, 0.30)
          • Garante clamp em [-1, 1]
        """
        mask = np.random.rand(self.vector_len) < self.mut_rate
        chrom.weights_vector[mask] += np.random.normal(0, 0.15, mask.sum())

        if random.random() < 0.30:
            idx = np.random.choice(
                self.vector_len, size=random.randint(1, 3), replace=False
            )
            chrom.weights_vector[idx] += np.random.normal(0, 0.30, len(idx))

        np.clip(chrom.weights_vector, -1, 1, out=chrom.weights_vector)

        if verbose:
            print(f"Total mutations: {mask.sum()}")

    def evolve(self, verbose: bool = False) -> np.ndarray:
        """Executa o GA e devolve o vetor de pesos do melhor cromossomo."""
        pop = self._init_pop()
        best_global: Chromosome | None = None

        for g in range(1, self.generations + 1):
            for c in pop:
                c.score = self.evaluator.evaluate(c.weights_vector)

            pop.sort(key=lambda c: c.score, reverse=True)

            if best_global is None or pop[0].score > best_global.score:
                best_global = pop[0].clone(keep_id=True)

            if verbose:
                print(f"Gen {g:>3}/{self.generations} | "
                    f"Best(gen) {pop[0].score:7.2f} | "
                    f"Best(ever) {best_global.score:7.2f}")

            # -------- Reprodução --------
            next_pop: List[Chromosome] = [pop[0].clone(keep_id=True)]  # elitismo

            while len(next_pop) < self.pop_size:
                p1 = self._select_tournament(pop)
                p2 = self._select_tournament(pop)
                while p2 is p1:
                    p2 = self._select_tournament(pop)

                child = self._crossover(p1, p2)

                if g > int(self.generations * 0.30):  # mutação só após 30 % das gerações
                    self._mutate(child)

                next_pop.append(child)

            pop = next_pop  # nova população

        if verbose:
            print("\nTreinamento concluído.\n")

        return best_global.weights_vector
