from math import inf

def minimax(current_board):
    """Algoritmo minimax que recebe um board e retorna a melhor posicao para o +1 jogar"""
    AI, HUMAN = +1, -1

    def check_winner(board, player):
        wins = [
            [board[0][0], board[0][1], board[0][2]],
            [board[1][0], board[1][1], board[1][2]],
            [board[2][0], board[2][1], board[2][2]],

            [board[0][0], board[1][0], board[2][0]],
            [board[0][1], board[1][1], board[2][1]],
            [board[0][2], board[1][2], board[2][2]],

            [board[0][0], board[1][1], board[2][2]],
            [board[2][0], board[1][1], board[0][2]],
        ]
        return [player, player, player] in wins

    def evaluate(board, depth):
        if check_winner(board, AI):   return +10 - depth
        if check_winner(board, HUMAN): return -10 + depth
        return 0

    def game_over(b):
        return check_winner(b, AI) or check_winner(b, HUMAN)

    def empty_cells(b):
        return [(i, j) for i in range(3) for j in range(3) if b[i][j] == 0]

    def _minimax(b, player, depth):
        moves = empty_cells(b)
        # terminal ou cheio
        if not moves or game_over(b):
            return -1, -1, evaluate(b, depth)

        if player == AI:
            best_score = -inf
            best_move = (-1, -1)
            for i, j in moves:
                b[i][j] = AI
                _, _, sc = _minimax(b, HUMAN, depth+1)
                b[i][j] = 0
                if sc > best_score:
                    best_score, best_move = sc, (i, j)
            return best_move[0], best_move[1], best_score

        else:  # HUMAN
            worst_score = +inf
            worst_move = (-1, -1)
            for i, j in moves:
                b[i][j] = HUMAN
                _, _, sc = _minimax(b, AI, depth+1)
                b[i][j] = 0
                if sc < worst_score:
                    worst_score, worst_move = sc, (i, j)
            return worst_move[0], worst_move[1], worst_score

    row, col, _ = _minimax([r.copy() for r in current_board], AI, depth=0)
    return row, col