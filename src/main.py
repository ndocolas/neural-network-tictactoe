from genetic.genetic_algorithm import run_evolution
from neural_network.network import NeuralNetwork
from game.ui import GameUI

if __name__ == "__main__":
    import sys

    generations = 100
    population_size = 50

    if len(sys.argv) == 3:
        generations = int(sys.argv[1])
        population_size = int(sys.argv[2])

    print("== Treinando rede neural com algoritmo genético ==")
    best_chromosome = run_evolution(generations=generations, population_size=population_size)
    nn = NeuralNetwork.from_chromosome(best_chromosome)

    print("== Iniciando interface gráfica: Rede Neural vs Minimax ==")
    game = GameUI(network=nn)
    game.run()
