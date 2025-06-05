from genetic.genetic_algorithm import run_evolution
from neural_network.network import NeuralNetwork
from game.front_end import run_game

if __name__ == "__main__":
    print("== Training neural network with heuristics for 100 games ==")
    best_chromosome = run_evolution(generations=100)
    network = NeuralNetwork.from_chromosome(best_chromosome)

    print("== Training finished. Play against the trained model ==")
    run_game(agent_type="neural_network", network=network)
