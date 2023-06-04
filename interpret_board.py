import cv2
import numpy as np
from tensorflow import keras

# Load the processed image generated in process_img.py in gray scale.
processed_image = cv2.imread('processed_image.jpg', 0)
processed_image = processed_image / 255.0
# Load the digit recognition model trained and saved in train_digit_recognition_model.py
digit_recognition_model = keras.models.load_model("trained_digit_model.h5")

# Size of each cell --> helps us split the grid into 9 rows & 9 cols
cell_size = processed_image.shape[0] // 9

cell_images = []

# Iterate over each row and col of the Sudoku grid (which is just an image at this point)
for row in range(9):
    for col in range(9):
        # Calculate top left corner coords of the cell
        x = col * cell_size
        y = row * cell_size

        # Scrape the cell image from the board
        cell_image = processed_image[y:y+cell_size, x:x+cell_size]
        cell_image = cv2.resize(cell_image, (28, 28))
        cell_image = cv2.threshold(cell_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        #cv2.imshow("Cell Image", cell_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        cell_images.append(cell_image)

cell_images = np.array(cell_images)

cell_images = cell_images.reshape(-1, 28, 28)

cell_images = cell_images / 255.0

#digit_probabilities = digit_recognition_model.predict(cell_images)

# Get the predicted digit labels
#digits = np.argmax(digit_probabilities, axis=1)

#print (digits)
# Use model to predict digit. 
# First line gives probabilities of each digit, so we get digit with the max probability 
#digit_probabilities = digit_recognition_model.predict(cell_image)
#digit = np.argmax(digit_probabilities)
#print (digit)
#print (", ")
# Store the digit in the matrix
#board_digits.append(digit)
#for i in board_digits:
 #   print (i)



