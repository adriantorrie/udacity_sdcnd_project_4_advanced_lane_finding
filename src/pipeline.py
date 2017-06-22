import cv2
import numpy as np


class Lane():
    """Receives the characteristics of each line detection"""
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None


def get_perspective_matrices():
    src_list = [(200, 720), (570, 470), (720, 470), (1130, 720)]
    dst_list = [(350, 720), (350, 0), (980, 0), (980, 720)]

    src = np.float32([src_list])
    dst = np.float32([dst_list])

    pers = cv2.getPerspectiveTransform(src, dst)
    pers_inv = cv2.getPerspectiveTransform(dst, src)

    return pers, pers_inv


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


def binary_threshold(img, thresh):
    """Applies threshold to an image to binarise the pixels

    Args:
        img (image): the image to transform

        thresh (tuple): (lower_bound, upper_bound)
                        lower_bound = lower bound for determining output
                        upper_bound = upper bound for determining output

    Returns:
        image: binary threshold image
    """
    binary_output = np.zeros_like(img)
    binary_output[(img > thresh[0]) & (img <= thresh[1])] = 1
    return binary_output


def abs_sobel_thresh(sobel, thresh=(20, 100)):
    """Return absolute sobel threshold image for further processing

    Args:
        sobel (image): the sobel image to apply the transform to

        thresh (tuple): (lower_bound, upper_bound)
                        lower_bound = lower bound for determining output
                        upper_bound = upper bound for determining output

    Returns:
        image: binary threshold image
    """
    sobel = np.absolute(sobel)
    sobel = np.uint8(255 * sobel / np.max(sobel))
    return binary_threshold(sobel, thresh)


def mag_thresh(x_sobel, y_sobel, thresh=(0, 255)):
    """Apply a threshold to the overall magnitude of the gradient

    Args:
        x_sobel (image): sobel image of gradients across the x axis

        y_sobel (image): sobel image of gradients across the y axis

        thresh (tuple): (lower_bound, upper_bound)
                        lower_bound = lower bound for determining output
                        upper_bound = upper bound for determining output

    Returns:
        image: binary threshold image
    """
    sobel = np.sqrt(x_sobel ** 2 + y_sobel ** 2)
    scale_factor = np.max(sobel) / 255
    sobel = (sobel / scale_factor).astype(np.uint8)
    return binary_threshold(sobel, thresh)


def dir_threshold(x_sobel, y_sobel, thresh=(0, np.pi / 4)):
    """Apply direction threshold

    Args:
        x_sobel (image): sobel image of gradients across the x axis

        y_sobel (image): sobel image of gradients across the y axis

        thresh (tuple): (lower_bound, upper_bound)
                        lower_bound = lower bound for determining output
                        upper_bound = upper bound for determining output

    Returns:
        image: binary threshold image
    """
    sobel = np.arctan2(np.absolute(y_sobel), np.absolute(x_sobel))
    return binary_threshold(sobel, thresh)


def sobel_binary(channel, ksize):
    """Combines the three sobel binary threshold images into a single image

    Sobel 1 = Absolute Sobel
    Sobel 2 = Magnitude Sobel
    Sobel 3 = Direction Sobel

    Args:
        channel (image): single channel image with 2D shape

        ksize (int): the size of the smoothing filter for the sobel operator

    Returns:
        image: binary threshold image
    """
    # set x and y sobels
    x_sobel = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=ksize)
    y_sobel = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=ksize)

    # get abs_sobel
    x_abs_sobel = abs_sobel_thresh(x_sobel, thresh=(20, 100))
    y_abs_sobel = abs_sobel_thresh(y_sobel, thresh=(20, 100))

    # get mag_sobel
    mag_sobel = mag_thresh(x_sobel, y_sobel, thresh=(20, 100))

    # get dir_sobel
    dir_sobel = dir_threshold(x_sobel, y_sobel, thresh=(0, np.pi / 4))

    # set sobel_binary
    sobel_binary = np.zeros_like(channel)
    sobel_binary[((x_abs_sobel == 1) & (y_abs_sobel == 1)) |
                 ((mag_sobel == 1) & (dir_sobel == 1))] = 1

    return sobel_binary


def colour_binary(rgb, hls, gray):
    """Combines three colour images into a single binary threshold image

    Args:
        rgb (image): image in RGB colour channel format (3D)

        hls (image): image in RGB colour channel format (3D)

        gray (image): image in grayscale colour channel format - (2D)

    Returns:
        image: binary threshold image
    """
    # combine colour binary thresholds
    r_binary = binary_threshold(rgb[:, :, 0], (220, 255))
    s_binary = binary_threshold(hls[:, :, 2], (170, 255))
    gray_binary = binary_threshold(gray, (220, 255))

    # set colour binary
    colour_binary = np.zeros_like(r_binary)
    colour_binary[((r_binary == 1) | (s_binary == 1) | (gray_binary == 1))] = 1

    return colour_binary


def combined_binary(img, ksize=3):
    """Create a binary threshold image from combining sobel and colour binary
    images.

    Args:
        img (image): the image to transform

        ksize (int): the size of the smoothing filter for the sobel operator

    Returns:
        image: binary threshold image
    """
    # set images
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # set binary threshold images
    colour_binary = colour_binary(gray, hls, rgb)
    sobel_binary = sobel_binary(hls[:, :, 1], ksize)

    # set combined binary
    combined_binary = np.zeros_like(r)
    combined_binary[((colour_binary == 1) | (sobel_binary == 1))] = 1

    return combined_binary


def warper(img, M):
    """Create a birds-eye view of the image

    Args:
        img (image): the image to transform

        M (np.array): the perspective matrix used to perform the transformation

    Returns:
        image: the warped image (birds-eye view)

    """
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


def pipeline(frame, M, dist, pers, pers_inv, left_lane, right_lane):
    """Annotates the frame with an overlay highlighting the lanes on a road

    Args:
        frame (image): the frame to annotate

        M (np.array): the camera matrix that will control warping

        dist (np.array): the distortion coefficients for warping

        pers (np.array): the perspective matrix for performing perspective
                         transform

        pers_inv (np.array): the inverse perspective matrix for restoring an
                             image back to its original perspective

        left_lane (Lane): used to manage annotations between frames

        right_lane (Lane): used to manage annotations between frames

    Returns:
        image: an image with lanes highlighted
    """
    img = undistort_img(frame, M, dist)
    img = combined_binary(img)
    img = warper(img, pers)

    # frame = frame + img

    return frame


if __name__ == "__main__":
    print('{} is to be imported not run directly'.format( __file__))
