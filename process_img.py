import cv2
import numpy as np

# At this stage, the image is in the same directory as this program
# So we can load the image using opencv's imread() function.

image = cv2.imread("sudokuimg2.jpg")

# Next, we need to apply some noise reduction techniques to this image.
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


cv2.drawContours(image, [sudoku_border], -1, (0, 255, 0), 2)

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
cv2.imwrite('processed_image.jpg', final_processed_image)

cv2.imshow("Sudoku Grid Detection", final_processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
