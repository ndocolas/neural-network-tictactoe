from src.minimax.minimax import minimax

class MinimaxPlayer:
    """
    Adapter para o algoritmo minimax.
    Responsabilidade única: escolher jogada com probabilidade de usar minimax.
    """

    def move(self, board: list[list[int]]) -> tuple[int, int]:
        """
        Retorna (linha, coluna). Com minimax, retorna o melhor lugar para jogar.
        """
        row, col = minimax(board)

        return row, col
