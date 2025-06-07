#!/usr/bin/env python3
from math import inf as infinity
from random import choice
import platform
import time
from os import system

HUMAN = -1
AI = +1

game_board = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]


def evaluate(board_state):
    if check_winner(board_state, AI):
        return +1
    elif check_winner(board_state, HUMAN):
        return -1
    else:
        return 0


# 50% random 50% minimax

def check_winner(board_state, player):
    win_conditions = [
        [board_state[0][0], board_state[0][1], board_state[0][2]],  # rows
        [board_state[1][0], board_state[1][1], board_state[1][2]],
        [board_state[2][0], board_state[2][1], board_state[2][2]],
        [board_state[0][0], board_state[1][0], board_state[2][0]],  # columns
        [board_state[0][1], board_state[1][1], board_state[2][1]],
        [board_state[0][2], board_state[1][2], board_state[2][2]],
        [board_state[0][0], board_state[1][1], board_state[2][2]],  # diagonals
        [board_state[2][0], board_state[1][1], board_state[0][2]],
    ]
    return [player, player, player] in win_conditions


def game_over(board_state):
    return check_winner(board_state, HUMAN) or check_winner(board_state, AI)


def empty_cells(board_state):
    empty_positions = []
    for row_index, row in enumerate(board_state):
        for col_index, cell in enumerate(row):
            if cell == 0:
                empty_positions.append([row_index, col_index])
    return empty_positions

#[0,1,2]
#[3,4,5]
#[6,7,8]

# 0, 0, 0
# 0, 1, 0
# 0, 0, 0

# H = 1
# IA = -1
# L = 0
def is_valid_move(row, col):
    return [row, col] in empty_cells(game_board)


def apply_move(row, col, player):
    if is_valid_move(row, col):
        game_board[row][col] = player
        return True
    return False


def minimax(board_state, available_moves_count, current_player):
    if current_player == AI:
        best_move = [-1, -1, -infinity]
    else:
        best_move = [-1, -1, +infinity]

    if available_moves_count == 0 or game_over(board_state):
        score = evaluate(board_state)
        return [-1, -1, score]

    for row, col in empty_cells(board_state):
        board_state[row][col] = current_player
        result = minimax(board_state, available_moves_count - 1, -current_player)
        board_state[row][col] = 0
        result[0], result[1] = row, col

        if current_player == AI and result[2] > best_move[2]:
            best_move = result
        elif current_player == HUMAN and result[2] < best_move[2]:
            best_move = result

    return best_move


def clear_console():
    os_name = platform.system().lower()
    if 'windows' in os_name:
        system('cls')
    else:
        system('clear')


def display_board(board_state, ai_symbol, human_symbol):
    symbols = {-1: human_symbol, +1: ai_symbol, 0: ' '}
    divider = f'{15 * "-"}'
    print('\n' + divider)
    for row in board_state:
        for cell in row:
            print(f'| {symbols[cell]} |', end='')
        print('\n' + divider)


def ai_turn(ai_symbol, human_symbol):
    available_moves = len(empty_cells(game_board))
    if available_moves == 0 or game_over(game_board):
        return

    clear_console()
    print(f'Computer turn [{ai_symbol}]')
    display_board(game_board, ai_symbol, human_symbol)

#[0,1,2] = 0,0 = 0 | 0,1 = 1 | 0, 2 = 2
#[3,4,5] = 1,0 = 3 | 1,1 = 4 | 1, 2 = 5
#[6,7,8] = 2,0 = 6 | 2,1 = 7 | 2, 2 = 8 

    if available_moves == 9:
        row = choice([0, 1, 2])
        col = choice([0, 1, 2])
    else:
        move = minimax(game_board, available_moves, AI)
        row, col = move[0], move[1]

    apply_move(row, col, AI)
    time.sleep(1)


def human_turn(ai_symbol, human_symbol):
    available_moves = len(empty_cells(game_board))
    if available_moves == 0 or game_over(game_board):
        return

    chosen_move = -1
    numpad_to_coords = {
        1: [0, 0], 2: [0, 1], 3: [0, 2],
        4: [1, 0], 5: [1, 1], 6: [1, 2],
        7: [2, 0], 8: [2, 1], 9: [2, 2],
    }

    clear_console()
    print(f'Human turn [{human_symbol}]')
    display_board(game_board, ai_symbol, human_symbol)

    while chosen_move < 1 or chosen_move > 9:
        try:
            chosen_move = int(input('Use numpad (1..9): '))
            selected_cell = numpad_to_coords[chosen_move]
            successful = apply_move(selected_cell[0], selected_cell[1], HUMAN)

            if not successful:
                print('Cell already occupied. Try again.')
                chosen_move = -1
        except (EOFError, KeyboardInterrupt):
            print('Bye')
            exit()
        except (KeyError, ValueError):
            print('Invalid input. Use a number between 1 and 9.')


def main():
    clear_console()
    human_symbol = ''
    ai_symbol = ''
    is_human_first = ''

    while human_symbol not in ['O', 'X']:
        try:
            human_symbol = input('Choose X or O\nChosen: ').upper()
        except (EOFError, KeyboardInterrupt):
            print('Bye')
            exit()
        except (KeyError, ValueError):
            print('Bad choice')

    ai_symbol = 'O' if human_symbol == 'X' else 'X'

    #ate aqui setou quem é X e quem é O

    clear_console()
    while is_human_first not in ['Y', 'N']:
        try:
            is_human_first = input('First to start? [Y/N]: ').upper()
        except (EOFError, KeyboardInterrupt):
            print('Bye')
            exit()
        except (KeyError, ValueError):
            print('Bad choice')

    while len(empty_cells(game_board)) > 0 and not game_over(game_board):
        if is_human_first == 'N':
            ai_turn(ai_symbol, human_symbol)
            is_human_first = ''

        human_turn(ai_symbol, human_symbol)
        ai_turn(ai_symbol, human_symbol)

    clear_console()
    display_board(game_board, ai_symbol, human_symbol)
    if check_winner(game_board, HUMAN):
        print('YOU WIN!')
    elif check_winner(game_board, AI):
        print('YOU LOSE!')
    else:
        print('DRAW!')
    exit()


if __name__ == '__main__':
    main()


# HUMANO = X
# IA = O


# def minimax(board)

# return place