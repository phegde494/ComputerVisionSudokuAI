import process_img
import solve_puzzle
import numpy as np
import cv2

def main():
    image = cv2.imread("images/onlinesudokuex.jpg")

    # Process image and get board as a numpy matrix
    board = process_img.getBoard(image)

    # Solve puzzle
    solved_board = solve_puzzle.solvePuzzle(board.tolist())

    if (solved_board):
        print (np.array(solved_board))
    else:
        print ("This sudoku cannot be solved")


if __name__ == "__main__":
    main()

