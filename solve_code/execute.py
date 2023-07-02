import solve_code.process_img as process_img
import solve_code.solve_puzzle as solve_puzzle
import numpy as np
import cv2

def main(file_path):
    image = cv2.imread(file_path)

    # Process image and get board as a numpy matrix
    board = process_img.getBoard(image)

    # Solve puzzle
    solved_board = solve_puzzle.solvePuzzle(board.tolist())

    if (solved_board):
        return solved_board
    else:
        print ("This sudoku cannot be solved\n\n")
    

