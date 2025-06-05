class MinimaxPlayer:
    def __init__(self, difficulty="medium"):
        self.difficulty = difficulty

    def get_move(self, game):
        for i, val in enumerate(game.get_board_state()):
            if val == 0:
                return i
        return -1
