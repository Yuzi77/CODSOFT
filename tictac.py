import tkinter as tk
from tkinter import messagebox
import math
import random

CELL_SIZE = 100
PADDING = 20
WIDTH = HEIGHT = CELL_SIZE * 3

class TicTacToeCanvas:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe (Canvas Version)")
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white")
        self.canvas.pack()

        self.board = [['' for _ in range(3)] for _ in range(3)]
        self.human_score = 0
        self.ai_score = 0
        self.draws = 0

        self.status = tk.Label(root, text=self.get_score_text(), font=("Arial", 14))
        self.status.pack()

        self.restart_btn = tk.Button(root, text="Restart", font=("Arial", 12), command=self.reset_game)
        self.restart_btn.pack(pady=5)

        self.canvas.bind("<Button-1>", self.on_click)
        self.draw_grid()

    def get_score_text(self):
        return f"Human (X): {self.human_score}  AI (O): {self.ai_score}  Draws: {self.draws}"

    def draw_grid(self):
        for i in range(1, 3):
            self.canvas.create_line(0, i * CELL_SIZE, WIDTH, i * CELL_SIZE, width=2)
            self.canvas.create_line(i * CELL_SIZE, 0, i * CELL_SIZE, HEIGHT, width=2)

    def draw_symbol(self, row, col, symbol):
        x1 = col * CELL_SIZE + PADDING
        y1 = row * CELL_SIZE + PADDING
        x2 = (col + 1) * CELL_SIZE - PADDING
        y2 = (row + 1) * CELL_SIZE - PADDING

        if symbol == 'X':
            self.canvas.create_line(x1, y1, x2, y2, fill='green', width=4)
            self.canvas.create_line(x1, y2, x2, y1, fill='green', width=4)
        elif symbol == 'O':
            self.canvas.create_oval(x1, y1, x2, y2, outline='blue', width=4)

    def draw_winning_line(self, line_type, index):
        if line_type == 'row':
            y = index * CELL_SIZE + CELL_SIZE // 2
            self.canvas.create_line(10, y, WIDTH - 10, y, fill="red", width=4)
        elif line_type == 'col':
            x = index * CELL_SIZE + CELL_SIZE // 2
            self.canvas.create_line(x, 10, x, HEIGHT - 10, fill="red", width=4)
        elif line_type == 'diag_main':
            self.canvas.create_line(10, 10, WIDTH - 10, HEIGHT - 10, fill="red", width=4)
        elif line_type == 'diag_anti':
            self.canvas.create_line(WIDTH - 10, 10, 10, HEIGHT - 10, fill="red", width=4)

    def on_click(self, event):
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        if self.board[row][col] == '' and not self.check_game_end():
            self.board[row][col] = 'X'
            self.draw_symbol(row, col, 'X')
            if not self.check_game_end():
                self.root.after(300, self.ai_turn)

    def ai_turn(self):
        move = self.best_move_human_can_win()
        if move:
            r, c = move
            self.board[r][c] = 'O'
            self.draw_symbol(r, c, 'O')
        self.check_game_end()

    def reset_game(self):
        self.board = [['' for _ in range(3)] for _ in range(3)]
        self.canvas.delete("all")
        self.draw_grid()

    def check_winner(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != '':
                return self.board[i][0], 'row', i
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != '':
                return self.board[0][i], 'col', i

        if self.board[0][0] == self.board[1][1] == self.board[2][2] != '':
            return self.board[0][0], 'diag_main', None

        if self.board[0][2] == self.board[1][1] == self.board[2][0] != '':
            return self.board[0][2], 'diag_anti', None

        if all(cell != '' for row in self.board for cell in row):
            return 'Draw', None, None

        return None, None, None

    def check_game_end(self):
        result, line_type, index = self.check_winner()
        if result:
            if result == 'Draw':
                self.draws += 1
                messagebox.showinfo("Game Over", "It's a draw!")
            elif result == 'X':
                self.human_score += 1
                self.draw_winning_line(line_type, index)
                messagebox.showinfo("Game Over", "You win!")
            elif result == 'O':
                self.ai_score += 1
                self.draw_winning_line(line_type, index)
                messagebox.showinfo("Game Over", "AI wins!")
            self.status.config(text=self.get_score_text())
            return True
        return False

    def best_move_human_can_win(self):
        # Make AI choose random available move to give human a fair chance
        empty = [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == '']
        return random.choice(empty) if empty else None

# --- Run the Game ---
if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToeCanvas(root)
    root.mainloop()
