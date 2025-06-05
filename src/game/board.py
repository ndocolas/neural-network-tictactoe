class Board:
    def __init__(self):
        self.board = [0] * 9
        self.current_player = 1

    def get_board_state(self):
        return self.board[:]

    def play_move(self, index: int):
        if 0 <= index < 9 and self.board[index] == 0:
            self.board[index] = self.current_player
            self.current_player *= -1
            return 1
        return -1

    def is_game_over(self):
        return all(cell != 0 for cell in self.board)

    def reset(self):
        self.board = [0] * 9
        self.current_player = 1
