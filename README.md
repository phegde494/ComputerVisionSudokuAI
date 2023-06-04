# ComputerVisionSudokuAI

Input an image of a sudoku puzzle (handwritten or a screenshot of your online game), and this AI will output a solution to the puzzle.

The AI uses image processing techniques such as greyscaling, Gaussian blurring, adaptive thresholding to reduce noise and enhance contour detection.

It then applies a four point perspective transform and scales the image appropriately. Next, it splits the processed image into a 9x9 grid of cells where each cell contains a digit.

Meanwhile, a digit classification model is trained on the MNIST data set (achieved 99.5% accuracy).

This model is then used to identify the digit (or lack of one) in each cell's image and place them into a 9x9 matrix of integers.

Finally, a sudoku solving algorithm utilizing backtracking is applied to the matrix, yielding a solution for the puzzle.
