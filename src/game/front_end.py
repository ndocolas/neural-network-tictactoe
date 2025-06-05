from game.board import Board
from minimax.minimax_player import MinimaxPlayer
from neural_network.network import NeuralNetwork

def run_game(agent_type="minimax", network: NeuralNetwork = None):
    game = Board()
    while not game.is_game_over():
        print("\n", game.get_board_state())
        if game.current_player == 1:
            idx = int(input("Sua jogada (0-8): "))
        else:
            if agent_type == "neural_network":
                output = network.forward(game.get_board_state())
                idx = max(range(9), key=lambda i: output[i] if game.get_board_state()[i] == 0 else -1)
            else:
                idx = MinimaxPlayer("medium").get_move(game)
        game.play_move(idx)

    print("\nFim de jogo. Estado final:", game.get_board_state())