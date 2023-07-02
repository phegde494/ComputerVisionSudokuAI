# Computer Vision Sudoku AI

**Input an image of a sudoku puzzle (handwritten or a screenshot of your online game), and this AI will output a solution to the puzzle.**

**Deep Learning & Computer Vision used to convert image to a 9x9 matrix of integers, and a backtracking algo is used to solve the puzzle.**

_______________________________
**Exact Methodology Below:**
__________________________

The web app allows the user to input an image, and the Flask server passes it along to the backend AI.

The AI uses image processing techniques such as greyscaling, Gaussian blurring, adaptive thresholding to reduce noise and enhance contour detection.

It then applies a four point perspective transform and scales the image appropriately. Next, it splits the processed image into a 9x9 grid of cells where each cell contains a digit.

Meanwhile, a convolutional neural network (CNN) is trained on the MNIST data set (achieved 99.5% accuracy).

This model is then used to identify the digit (or lack of one) in each cell's image and place them into a 9x9 matrix of integers.

Finally, a sudoku solving algorithm utilizing backtracking is applied to the matrix, yielding a solution for the puzzle. The web app displays the solution.

**See demo below:**




https://github.com/phegde494/ComputerVisionSudokuAI/assets/48624928/3f41f34c-6799-4cef-8659-728a835feae6

