import csv
import os
import random
from pathlib import Path
from typing import List

import numpy as np

from entities.chromosome import Chromosome
from usecases.score_evaluator import ScoreEvaluator


class GeneticAlgorithm:
    """
    Algoritmo Genético com:
      • Avaliação por n_games (80 % p=0.5, 20 % p=1.0)
      • Critério de parada por número de gerações
      • Elitismo (melhor indivíduo segue intacto, preservando o mesmo ID)
      • CSV por geração no formato IA,Score
    """

    def __init__(
        self,
        population_size: int,
        generations: int,
        n_games: int,
        out_dir: str | os.PathLike = "populations",
    ):
        # hiperparâmetros principais
        self.pop_size = population_size
        self.generations = generations
        self.n_games = n_games

        # arquitetura da rede (fixa: 9-9-9)
        self.in_size = self.h_size = self.o_size = 9
        self.vector_len = 180

        # GA
        self.mut_rate = 0.1
        self.evaluator = ScoreEvaluator(
            self.in_size, self.h_size, self.o_size, n_games
        )

        # pasta de saída dos CSVs
        self.out_path = Path(out_dir)
        self.out_path.mkdir(parents=True, exist_ok=True)

        # reinicia contagem global de IDs (opcional)
        # Chromosome._next_id = 0

    def _init_pop(self) -> List[Chromosome]:
        return [
            Chromosome(np.random.uniform(-1, 1, self.vector_len))
            for _ in range(self.pop_size)
        ]

    def _select_tournament(self, pop: List[Chromosome]) -> Chromosome:
        """Retorna o melhor entre *k* indivíduos sorteados."""
        return max(random.sample(pop, 2), key=lambda c: c.score)

    def _crossover(self, father: Chromosome, mother: Chromosome):
        child_vec = np.zeros(self.vector_len)
        for i in range(self.vector_len):
            child_vec[i] = (father.weights_vector[i] + mother.weights_vector[i]) / 2.0
        return Chromosome(child_vec)

    def _mutate(self, chrom: Chromosome, verbose: bool = False) -> Chromosome:
        """
        Mutação real-coded:
          • perturba ~= mut_rate dos genes com N(0, 0.15)
          • 30 % de chance de burst extra (1-3 genes) com N(0, 0.30)
          • garante faixa [-1, 1]
        (Modifica in-place — ID preservado)
        """
        mask = np.random.rand(self.vector_len) < self.mut_rate
        chrom.weights_vector[mask] += np.random.normal(0, 0.15, mask.sum())

        if random.random() < 0.3:  # burst extra
            idx = np.random.choice(
                self.vector_len, size=random.randint(1, 3), replace=False
            )
            chrom.weights_vector[idx] += np.random.normal(0, 0.30, len(idx))

        np.clip(chrom.weights_vector, -1, 1, out=chrom.weights_vector)

        if verbose:
            print(f"Total mutations: {mask.sum()}")

        return chrom

    # ------------------------------------------------------------------ #
    # ------------------------ CSV UTIL -------------------------------- #
    # ------------------------------------------------------------------ #
    def _save_population_csv(self, generation: int, pop: List[Chromosome]) -> None:
        """Grava CSV `population_gen{generation}.csv` com colunas IA,Score."""
        file_path = self.out_path / f"population_gen{generation}.csv"
        with file_path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["IA", "Score"])
            for chrom in pop:
                writer.writerow([chrom.id, f"{chrom.score:.4f}"])

    # ------------------------------------------------------------------ #
    # --------------------------- EVOLVE ------------------------------- #
    # ------------------------------------------------------------------ #
    def evolve(self, verbose: bool = False) -> np.ndarray:
        pop = self._init_pop()
        best_chrom: Chromosome | None = None
        best_fit = -float("inf")

        for g in range(1, self.generations + 1):
            # ----------- Avaliação -----------
            for c in pop:
                c.score = self.evaluator.evaluate(c.weights_vector)

            # ordena descendente por score
            pop.sort(key=lambda c: c.score, reverse=True)

            # CSV da geração atual
            self._save_population_csv(g, pop)

            # atualiza melhor global
            if pop[0].score > best_fit:
                best_fit = pop[0].score
                best_chrom = pop[0]

            # logging
            if verbose:
                avg_fit = sum(c.score for c in pop) / len(pop)
                print(
                    f"Gen {g:>3}/{self.generations} | "
                    f"Best(gen): {pop[0].score:>7.2f} | "
                    f"Best(ever): {best_fit:>7.2f}"
                )

            # ----------- Reprodução -----------
            best = pop[0]
            if verbose:
                print(f"{best}")
            next_pop: List[Chromosome] = [best]  # elitismo (mesma instância)

            while len(next_pop) < self.pop_size:
                p1 = self._select_tournament(pop)
                p2 = self._select_tournament(pop)
                while p2 is p1:
                    p2 = self._select_tournament(pop)

                child = self._crossover(p1, p2)
                if g > round(self.generations * 0.3):
                    self._mutate(child)
                next_pop.append(child)

            pop = next_pop  # nova população (sem score ainda)

        if verbose:
            print("\nTreinamento completo.\n")

        # garante retorno mesmo se nenhuma melhoria após g=1
        return (best_chrom or pop[0]).weights_vector
