import cv2
from cv2 import Canny
import numpy as np


class LaneLines:

    def make_coordinates(image, line_parameters):
        slope, intercept = line_parameters
        y1 = image.shape[0]  # Height
        y2 = int(y1 * (3/5))
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return np.array([x1, y1, x2, y2])

    def average_slope_intercept(image, lines):
        left_fit = []
        right_fit = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = LaneLines.make_coordinates(image, left_fit_average)
        right_line = LaneLines.make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])

    def canny(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = 5
        blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        canny = cv2.Canny(blur, 50, 150)
        return canny

    def region_of_interest(img):
        height = img.shape[0]
        width = img.shape[1]
        mask = np.zeros_like(img)
        triangle = np.array(
            [[(200, height), (800, 350), (1200, height), ]], np.int32)
        cv2.fillPoly(mask, triangle, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def houghLines(img):
        houghLines = cv2.HoughLinesP(
            img, 2, np.pi/180, 100, minLineLength=30, maxLineGap=5)
        return houghLines

    def display_lines(img, lines):
        line_image = np.zeros_like(img)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
        return line_image

    def forward(data):
        img = data
        this = LaneLines
        canny_output = this.canny(img)
        masked_output = this.region_of_interest(canny_output)
        lines = this.houghLines(masked_output)
        averaged_lines = this.average_slope_intercept(img, lines)
        line_image = this.display_lines(img, averaged_lines)
        combo_image = cv2.addWeighted(img, 0.8, line_image, 1, 1)
        return combo_image
