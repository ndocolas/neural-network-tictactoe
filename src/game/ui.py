import tkinter as tk
from functools import partial
from game.board import Board
from neural_network.network import NeuralNetwork
from minimax.minimax_player import MinimaxPlayer

class GameUI:
    def __init__(self, network: NeuralNetwork):
        self.root = tk.Tk()
        self.root.title("Jogo da Velha: Rede Neural vs Minimax")
        self.buttons = []
        self.board = Board()
        self.network = network
        self.minimax = MinimaxPlayer("medium")
        self.build_ui()
        self.current_turn = "network"
        self.root.after(500, self.play_turn)

    def build_ui(self):
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                button = tk.Button(self.root, text="", width=10, height=4, state="disabled")
                button.grid(row=i, column=j)
                self.buttons.append(button)

    def play_turn(self):
        if self.board.is_game_over():
            self.update_buttons()
            return

        if self.current_turn == "network":
            self.play_network()
            self.current_turn = "minimax"
        else:
            self.play_minimax()
            self.current_turn = "network"

        self.update_buttons()
        if not self.board.is_game_over():
            self.root.after(500, self.play_turn)

    def play_network(self):
        state = self.board.get_board_state()
        output = self.network.forward(state)
        move = max(range(9), key=lambda i: output[i] if state[i] == 0 else -1)
        self.board.play_move(move)

    def play_minimax(self):
        move = self.minimax.get_move(self.board)
        self.board.play_move(move)

    def update_buttons(self):
        for i, val in enumerate(self.board.get_board_state()):
            if val == 1:
                self.buttons[i]["text"] = "X"
                self.buttons[i]["state"] = "disabled"
            elif val == -1:
                self.buttons[i]["text"] = "O"
                self.buttons[i]["state"] = "disabled"

        if self.board.is_game_over():
            for b in self.buttons:
                b["state"] = "disabled"

    def run(self):
        self.root.mainloop()
