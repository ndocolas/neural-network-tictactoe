from usecases.score_evaluator import ScoreEvaluator
from entities.chromosome import Chromosome
from pathlib import Path
from typing import List
import numpy as np
import random
import csv
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
    ):
        # ----- Hiperparâmetros principais -----
        self.pop_size = population_size
        self.generations = generations
        self.n_games = n_games

        # Arquitetura fixa 9-9-9 → 180 pesos
        self.in_size = self.h_size = self.o_size = 9
        self.vector_len = 180

        # Taxas do GA
        self.mut_rate = 0.20

        self.out_path = Path("populations")
        self.out_path.mkdir(parents=True, exist_ok=True)

        # Avaliador de fitness
        self.evaluator = ScoreEvaluator(
            self.in_size, self.h_size, self.o_size, n_games
        )

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

    def _save_population_csv(self, generation: int, pop: List[Chromosome]) -> None:
        """Grava CSV `population_gen{generation}.csv` com colunas IA,Score."""
        file_path = self.out_path / f"population_gen{generation}.csv"
        with file_path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["IA", "Score"])
            for chrom in pop:
                writer.writerow([chrom.id, f"{chrom.score:.4f}"])

    def _mutate(self, chrom: Chromosome, verbose: bool = False) -> None:
        """
        Mutação real-coded in-place ‒ versão simples:

          • Cada gene tem prob. `mut_rate` de ser mutado.
          • Genes mutados recebem NOVO valor U(-1, 1) — não é perturbação.
          • (Opcional) “burst” extra: 30 % de chance de ainda trocar 1–3 genes.
          • Após a troca, o vetor já está garantidamente dentro de [-1, 1].
        """
        # ---- mutação normal ------------------------------------------------
        mask = np.random.rand(self.vector_len) < self.mut_rate
        num_mut = mask.sum()
        if num_mut:
            chrom.weights_vector[mask] = np.random.uniform(-1, 1, num_mut)

        # ---- burst opcional ------------------------------------------------
        if random.random() < 0.30:
            extra = np.random.choice(
                self.vector_len, size=random.randint(1, 3), replace=False
            )
            chrom.weights_vector[extra] = np.random.uniform(-1, 1, extra.size)
            num_mut += extra.size

        if verbose:
            print(f"Total genes mutated: {num_mut}")


    def evolve(self, verbose: bool = False) -> np.ndarray:
        """Executa o GA e devolve o vetor de pesos do melhor cromossomo."""
        pop = self._init_pop()
        best_global: Chromosome | None = None

        for g in range(1, self.generations + 1):
            for c in pop:
                c.score = self.evaluator.evaluate(c.weights_vector)

            pop.sort(key=lambda c: c.score, reverse=True)
            self._save_population_csv(g, pop)

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
