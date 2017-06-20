import cv2
import numpy as np

def undistort_img(img, M, dist):
    """Un-distorts image on disk

    Args:
        img_path (str): Path to image to undistort

        M (numpy.array): Camera matrix (output from cv2.calibrateCamera())

        dist (numpy.array): Distortion coefficients (output from
                            cv2.calibrateCamera())

    Returns
        image: numpy.array representation of an image

    """
    return cv2.undistort(img, M, dist, None, M)


def abs_sobel_thresh(img, orient='x', thresh_min=50, thresh_max=100):
    """Return absolute sobel threshold image for further processing

    Steps taken
        1) Convert to grayscale
        2) Take the derivative in x or y given orient = 'x' or 'y'
        3) Take the absolute value of the derivative or gradient
        4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        5) Create a mask of 1's where the scaled gradient magnitude
                is > thresh_min and < thresh_max

    Args:
        img (image): the image to apply the transform to

        orient (str): One of ['x', 'y'] axis to apply sobel operator to

        thresh_min (int): lower bound for determining binary output

        thresh_max (int`):: upper bound for determining binary output

    Returns:
        image: binary threshold image
    """

    # Define a function that takes an image, gradient orientation,
    # and threshold min / max values.

    # 1
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2 & 3
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    # 4
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # 5
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[
        (scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """Apply a threshold to the overall magnitude of the gradient

    Steps taken
        1) Convert to grayscale
        2) Take the gradient in x and y separately
        3) Calculate the magnitude
        4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        5) Create a binary mask where mag thresholds are met

    Args:
        img (image): the image to apply the transform to

        sobel_kernel (int): (default=3) region to apply magnitude over

        mag_thresh (tuple): (lower_bound, upper_bound)
                            lower_bound = lower bound for determining output
                            upper_bound = upper bound for determining output

    Returns:
        image: binary threshold image
    """
    # 1
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 4
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)

    # 5
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """
    Apply the following steps to img
        1) Convert to grayscale
        2) Take the gradient in x and y separately
        3) Take the absolute value of the x and y gradients
        4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        5) Create a binary mask where direction thresholds are met

    Args:
        img (image): the image to apply the transform to

        sobel_kernel (int): (default=3) region to apply magnitude over

        thresh (tuple): (lower_bound, upper_bound)
                        lower_bound = lower bound for determining output
                        upper_bound = upper bound for determining output

    Returns:
        image: binary threshold image
    """
    # 1
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3 & 4
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # 5
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output


def binary_threshold():

    return binary_output


def get_perspective_matrices():
    src_list = [(200, 720), (570, 470), (720, 470), (1130, 720)]
    dst_list = [(350, 720), (350, 0), (980, 0), (980, 720)]

    src = np.float32([src_list])
    dst = np.float32([dst_list])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return M, M_inv


def warper(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped


def run_pipeline(img):
    # locals
    M, M_inv = get_perspective_matrices()

    # pipeline
    img = undistort_img(img)
    img = binary_threshold(img)
    img = warper(img, M)

    return img


if __name__ == "__main__":
    print('{} is to be imported not run directly'.format( __file__))
