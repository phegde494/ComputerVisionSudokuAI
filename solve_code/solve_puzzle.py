
# Prints solved board or says it isn't possible
def solvePuzzle(board):
    res = solve(board)

    if (res):
        return board
    else:
        return None

# Solves a sudoku puzzle
def solve(board):
    if not getEmptyCell(board):
        return True
    
    row, col = getEmptyCell(board)
    
    # Loops through each possible value for that cell
    for val in range(1, 10):
        if validSudoku(board, row, col, val):
            board[row][col] = val
            # Continuing with this decision, if it's possible to solve the puzzle, we return true
            # Since we mutate the board as we go, it'll already be solved when we return
            if solve(board):
                return True
            
            # Otherwise, we reset this cell and try again with the next possible value
            board[row][col] = 0
    
    # If we broke out of the loop, that means none of the possible values worked, so we cannot solve the puzzle
    return False

# Loops from top left to bottom right and returns the first empty cell present in the board
def getEmptyCell(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return i, j
    return None

# Returns whether the current board is valid by sudoku rules
def validSudoku(board, row, col, val):
    # Check each element in the same row as this val
    for i in range(9):
        if board[row][i] == val:
            return False
    
    # Check each element in the same column as this val
    for i in range(9):
        if board[i][col] == val:
            return False
    
    # Check each element in the same 3x3 grid as this val

    # Row & col of the top left element of this 3x3 grid
    top_left_row = (row // 3) * 3   
    top_left_col = (col // 3) * 3
    
    for i in range(3):
        for j in range(3):
            if board[top_left_row + i][top_left_col + j] == val:
                return False
    
    return True
