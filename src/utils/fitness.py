from typing import List
from neural_network.network import NeuralNetwork
from game.board import Board
from minimax.minimax_player import MinimaxPlayer
import random

CONVERGENCE_THRESHOLD = 0.001
CONVERGENCE_COUNT = 10

def calculate_fitness(network: NeuralNetwork) -> float:
    score = 0
    game = Board()
    minimax = MinimaxPlayer(difficulty="hard")

    for _ in range(5):
        game.reset()
        invalid_moves = 0
        winner = None

        while not game.is_game_over():
            state = game.get_board_state()
            current_player = game.current_player

            if current_player == 1:
                output = network.forward(state)
                move = max(range(9), key=lambda i: output[i] if state[i] == 0 else -1)
            else:
                move = minimax.get_move(game)

            result = game.play_move(move)
            if result == -1 and current_player == 1:
                score -= 5
                invalid_moves += 1
                break

        final_state = game.get_board_state()
        if all(v != 0 for v in final_state):
            score += 1

        if winner := detect_winner(final_state):
            if winner == 1:
                score += 5
            elif winner == -1:
                score -= 2

        if invalid_moves > 0:
            score -= invalid_moves * 2

    return score

def detect_winner(state: List[int]) -> int:
    wins = [
        [0,1,2], [3,4,5], [6,7,8],
        [0,3,6], [1,4,7], [2,5,8],
        [0,4,8], [2,4,6]
    ]
    for a, b, c in wins:
        if state[a] == state[b] == state[c] != 0:
            return state[a]
    return 0

def tournament_selection(population: List[List[float]], fitnesses: List[float], k: int = 3) -> List[float]:
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]

def evaluate_population(population: List[List[float]]) -> List[float]:
    return [calculate_fitness(NeuralNetwork.from_chromosome(chrom)) for chrom in population]

def check_convergence(fitness_history: List[float]) -> bool:
    if len(fitness_history) < CONVERGENCE_COUNT:
        return False
    deltas = [abs(fitness_history[-i] - fitness_history[-i - 1]) for i in range(1, CONVERGENCE_COUNT)]
    return all(delta < CONVERGENCE_THRESHOLD for delta in deltas)

def test_accuracy(network: NeuralNetwork, rounds: int = 20) -> float:
    wins = 0
    ties = 0
    losses = 0
    game = Board()
    minimax = MinimaxPlayer(difficulty="hard")

    for _ in range(rounds):
        game.reset()
        while not game.is_game_over():
            state = game.get_board_state()
            current_player = game.current_player

            if current_player == 1:
                output = network.forward(state)
                move = max(range(9), key=lambda i: output[i] if state[i] == 0 else -1)
            else:
                move = minimax.get_move(game)

            game.play_move(move)

        winner = detect_winner(game.get_board_state())
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            ties += 1

    accuracy = wins / rounds
    print(f"Rede Neural - Vitórias: {wins}, Empates: {ties}, Derrotas: {losses}, Acurácia: {accuracy:.2%}")
    return accuracy
