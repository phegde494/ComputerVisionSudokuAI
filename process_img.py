import cv2
import numpy as np
from tensorflow import keras

def main():
    # At this stage, the image is in the same directory as this program
    # So we can load the image using opencv's imread() function.
    image = cv2.imread("images/onlinesudokuex.jpg")
    board = getBoard(image)

    print(board)

def getBoard(image):
    
    final_processed_image = processImage(image)

    # Then we identify each of the digits in the board and place them into a matrix
    board = getBoardDigits(final_processed_image)

    return board

# Here, we process the image so that only the board is present, 
# in a format from which individual cells can be isolated
# Returns this processed image
def processImage(image):
    contours = findContours(image)
    sudoku_border = identifyCorrectBorder(contours)

    cv2.drawContours(image, [sudoku_border], -1, (0, 255, 0), 2)

    final_processed_image = transformImage(image, sudoku_border)

    cv2.imshow("Sudoku Grid Detection", final_processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    return final_processed_image


# Returns a list of all contours in the image
def findContours(image):
    # We need to apply some noise reduction techniques to this image.
    # This is necessary in order for the contour-finding algorithm applied in later steps to be effective.

    # We can first convert the image to grayscale. 
    # Normally, the pixels would have three channels (RGB), but by applying this technique, we now only have 1 channel.
    # Additionally, this helps eliminate any issues caused by color variations and lighting conditions.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Next, we can blur the image using a Gaussian blur algorithm
    # This helps reduce noise by smoothing the image and blending each pixel with its neighboring pixel values.
    # This greatly improves the performance of the adaptive thresholding algo.
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 3)

    # Adaptive thresholding allows us create better contrast between the sudoku board cells and their neighboring pixels.
    # First, it creates a threshold value for each pixel (weighted sum of pixels in local neighborhood via Gaussian distribution)
    # Then, it gives each pixel a binary value depending on whether or not it's greater than the threshold.
    # Now, everything is either black or white, creating contrast that facilitates processing

    adaptive_threshold = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Now, the image is ready to be scanned using a contour detection algorithm
    # Each detected contour is stored in a list
    contours, _ = cv2.findContours(adaptive_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

# Identifies the appropriate contour given a list of all contours in the image
def identifyCorrectBorder(contours):
    sudoku_border = None
    max_area = 0

    # Here, we loop through each contour and choose the one with the largest area which also happens to have four sides (quadrilateral)
    # This is the contour most likely to be our sudoku border.
    # Choosing the largest contour helps prevent issues where contours exist around individual board cells.
    for c in contours:
        contour_perimeter_arc = cv2.arcLength(c, True)
        sides = cv2.approxPolyDP(c, 0.02 * contour_perimeter_arc, True)

        if (len(sides) == 4 and cv2.contourArea(c) > max_area):
            max_area = cv2.contourArea(c)
            sudoku_border = c

    # If we never identified a contour which passed the required criteria, we throw an exception.
    if sudoku_border is None:
        raise Exception("Unable to find the border for this Sudoku. Please take a clearer picture.")
    
    return sudoku_border

# Transforms and returns the image
def transformImage(image, sudoku_border):
    # Next, we want to transform the image so that it is parallel to the page, making it easy to process cells.
    # We can achieve this using a four point perspective transform.
    epsilon = 0.1 * cv2.arcLength(sudoku_border, True)
    approx_corners = cv2.approxPolyDP(sudoku_border, epsilon, True)

    # Here, we reorder the points so that they are in a clockwise order. This helps avoid any flipped/rotated images.
    corner_points = np.zeros((4, 2), dtype=np.float32)
    sums = approx_corners.sum(axis=2)
    corner_points[0] = approx_corners[np.argmin(sums)]
    corner_points[2] = approx_corners[np.argmax(sums)]

    diffs = np.diff(approx_corners, axis=2)
    corner_points[1] = approx_corners[np.argmin(diffs)]
    corner_points[3] = approx_corners[np.argmax(diffs)]

    # We want the transformed image to be 450x450 pixels, so we define an array with each corner point.
    target_points = np.float32([[0, 0], [450, 0], [450, 450], [0, 450]])

    # Compute perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(corner_points, target_points)


    # Apply the perspective transform to the original image using the matrix and the specified dimensions
    final_processed_image = cv2.warpPerspective(image, perspective_matrix, (450, 450))

    final_processed_image = cv2.cvtColor(final_processed_image, cv2.COLOR_BGR2GRAY)

    final_processed_image = cv2.threshold(final_processed_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    return final_processed_image


# Returns a matrix representing the board shown in the image
def getBoardDigits(final_processed_image):

    # Load the digit recognition model trained and saved in train_digit_recognition_model.py
    digit_recognition_model = keras.models.load_model("model/trained_digit_model.h5")

    # Size of each cell --> helps us split the grid into 9 rows & 9 cols
    cell_size = final_processed_image.shape[0] // 9

    digits = []

    # Iterate over each row and col of the Sudoku grid (which is just an image at this point)
    for row in range(9):
        for col in range(9):
            cell_image = getCellImageAt(row, col, cell_size, final_processed_image)
            
            cell_image = reformatCellImage(cell_image)

            digit = predictDigit(digit_recognition_model, cell_image)
            digits.append(digit)
    

    digits = np.array(digits)

    return digits


# Isolates the individual cell at the appropriate row & col (board is 9x9)
# Returns the image of that cell
def getCellImageAt(row, col, cell_size, final_processed_image):
    # Calculate top left corner coords of the cell
    x = col * cell_size
    y = row * cell_size

    # Scrape the cell image from the board
    cell_image = final_processed_image[y:y+cell_size, x:x+cell_size]

    # Calculate the border thickness as percent of image size
    border_percentage = 0.1  # 1/10th of the image size
    border_thickness = int(min(cell_image.shape[:2]) * border_percentage)

    x = border_thickness
    y = border_thickness
    width = cell_image.shape[1] - 2 * border_thickness
    height = cell_image.shape[0] - 2 * border_thickness

    # Crop the image to remove the four borders. 
    # This is because they'll have traces of white (originally black) in them and will mess with the model
    cell_image = cell_image[y:y+height, x:x+width]

    return cell_image

# Reformats the cell image such that it is appropriate for the model
# As close as possible to images in MNIST dataset
# Returns the image after changes have been made
def reformatCellImage(cell_image):
    # Make cell image 28x28 with each pixel value from 0 to 1
    cell_image = cv2.resize(cell_image, (28, 28))
    cell_image = cell_image.astype('float32') / 255.0
    cell_image = np.reshape(cell_image, (1, 28, 28, 1))

    return cell_image

# Use our trained neural network to predict the digit in the cell.
# Returns the predicted digit
def predictDigit(digit_recognition_model, cell_image):
    digit = 0
    # The model only works if there actually is a digit.
    # As long as the average pixel value in the image is above 0.05, it's likely there's a digit (0 = black, 1 = white)
    if (np.average(cell_image) > 0.05):
        digit_probabilities = digit_recognition_model.predict(cell_image)
        digit = np.argmax(digit_probabilities)
    
    return digit

    

if __name__ == "__main__":
    main()
