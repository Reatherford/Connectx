import numpy as np
import random

class Connect4:
    def __init__(self, rows=6, columns=7):
        self.rows = rows
        self.columns = columns
        self.board = np.zeros((rows, columns), dtype=int)
        self.current_player = 1

    def drop_piece(self, column):
        for row in range(self.rows-1, -1, -1):
            if self.board[row][column] == 0:
                self.board[row][column] = self.current_player
                return True
        return False

    def is_valid_move(self, column):
        return 0 <= column < self.columns and self.board[0][column] == 0

    def check_winner(self):
        # Check horizontal
        for row in range(self.rows):
            for col in range(self.columns - 3):
                if self.board[row][col] == self.board[row][col+1] == self.board[row][col+2] == self.board[row][col+3] != 0:
                    return self.board[row][col]

        # Check vertical
        for row in range(self.rows - 3):
            for col in range(self.columns):
                if self.board[row][col] == self.board[row+1][col] == self.board[row+2][col] == self.board[row+3][col] != 0:
                    return self.board[row][col]

        # Check diagonal (top-left to bottom-right)
        for row in range(self.rows - 3):
            for col in range(self.columns - 3):
                if self.board[row][col] == self.board[row+1][col+1] == self.board[row+2][col+2] == self.board[row+3][col+3] != 0:
                    return self.board[row][col]

        # Check diagonal (top-right to bottom-left)
        for row in range(self.rows - 3):
            for col in range(3, self.columns):
                if self.board[row][col] == self.board[row+1][col-1] == self.board[row+2][col-2] == self.board[row+3][col-3] != 0:
                    return self.board[row][col]

        # Check for draw
        if np.all(self.board != 0):
            return 0

        return None

    def switch_player(self):
        self.current_player = 3 - self.current_player

    def print_board(self):
        print(np.flip(self.board, 0))

class AI:
    def __init__(self, player):
        self.player = player

    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = 1 if piece == 2 else 2

        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(0) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(0) == 2:
            score += 2

        if window.count(opp_piece) == 3 and window.count(0) == 1:
            score -= 4

        return score

    def score_position(self, board):
        score = 0

        # Score center column
        center_array = [int(i) for i in list(board[:, 3])]
        center_count = center_array.count(self.player)
        score += center_count * 3

        # Score horizontal
        for r in range(6):
            row_array = [int(i) for i in list(board[r,:])]
            for c in range(4):
                window = row_array[c:c+4]
                score += self.evaluate_window(window, self.player)

        # Score vertical
        for c in range(7):
            col_array = [int(i) for i in list(board[:,c])]
            for r in range(3):
                window = col_array[r:r+4]
                score += self.evaluate_window(window, self.player)

        # Score diagonal
        for r in range(3):
            for c in range(4):
                window = [board[r+i][c+i] for i in range(4)]
                score += self.evaluate_window(window, self.player)

        for r in range(3):
            for c in range(4):
                window = [board[r+3-i][c+i] for i in range(4)]
                score += self.evaluate_window(window, self.player)

        return score

    def get_valid_locations(self, board):
        return [col for col in range(7) if board[0][col] == 0]

    def is_terminal_node(self, game):
        return game.check_winner() is not None or len(self.get_valid_locations(game.board)) == 0

    def minimax(self, game, depth, alpha, beta, maximizing_player):
        valid_locations = self.get_valid_locations(game.board)
        is_terminal = self.is_terminal_node(game)
        if depth == 0 or is_terminal:
            if is_terminal:
                winner = game.check_winner()
                if winner == self.player:
                    return (None, 100000000000000)
                elif winner == 3 - self.player:
                    return (None, -10000000000000)
                else:  # Game is over, no more valid moves
                    return (None, 0)
            else:  # Depth is zero
                return (None, self.score_position(game.board))
        if maximizing_player:
            value = -np.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                game_copy = Connect4()
                game_copy.board = game.board.copy()
                game_copy.drop_piece(col)
                new_score = self.minimax(game_copy, depth-1, alpha, beta, False)[1]
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column, value

        else:  # Minimizing player
            value = np.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                game_copy = Connect4()
                game_copy.board = game.board.copy()
                game_copy.current_player = 3 - self.player
                game_copy.drop_piece(col)
                new_score = self.minimax(game_copy, depth-1, alpha, beta, True)[1]
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value

    def get_best_move(self, game, depth=4):
        column, _ = self.minimax(game, depth, -np.inf, np.inf, True)
        return column

def play_game():
    game = Connect4()
    ai = AI(2)  # AI plays as player 2

    while True:
        game.print_board()
        if game.current_player == 1:
            while True:
                try:
                    col = int(input(f"Player {game.current_player}, choose a column (0-6): "))
                    if game.is_valid_move(col):
                        game.drop_piece(col)
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Please enter a valid integer between 0 and 6.")
        else:
            col = ai.get_best_move(game)
            game.drop_piece(col)
            print(f"AI dropped piece in column {col}")

        winner = game.check_winner()
        if winner is not None:
            game.print_board()
            if winner == 0:
                print("It's a draw!")
            else:
                print(f"Player {winner} wins!")
            break

        game.switch_player()

if __name__ == "__main__":
    play_game()