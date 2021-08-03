import cv2 as cv


def process(gray_image, low_threshold=0, ratio=3, kernel_size=1):
    detected_edges = cv.Canny(gray_image, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    dst = gray_image * (mask.astype(gray_image.dtype))
    return dst, detected_edges
