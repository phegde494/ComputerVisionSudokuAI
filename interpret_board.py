import cv2
import numpy as np
from tensorflow import keras
import imageio
# Load the processed image generated in process_img.py in gray scale.
processed_image = imageio.imread('processed_image.jpg')
processed_image = processed_image / 255.0
processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
print (processed_image.shape)
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
        print (cell_image.shape)
        cell_image = cv2.resize(cell_image, (28, 28))
        print (cell_image.shape)
        #min_value = np.min(cell_image)
        #max_value = np.max(cell_image)


        #converted_image = cv2.normalize(cell_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        #cell_image = cv2.cvtColor(cell_image, cv2.COLOR_RGB2GRAY)
        #print(cell_image.shape)
        #cell_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        #cell_image = cv2.adaptiveThreshold(cell_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        #cell_image = cell_image.reshape(1, 28, 28)


        #digit_probabilities = digit_recognition_model.predict(cell_image)
        #digit = np.argmax(digit_probabilities)
        #print (digit)
        #cell_image = cell_image.reshape(28, 28, 1)
        cv2.imshow("Cell Image", cell_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cell_images.append(cell_image)

cell_images = np.array(cell_images)

#cell_images = cell_images.reshape(-1, 28, 28)

#cell_images = cell_images / 255.0

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



