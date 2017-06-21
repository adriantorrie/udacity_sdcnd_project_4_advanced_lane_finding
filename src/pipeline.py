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


def binary_threshold(img, thresh):
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


# def mag_thresh(x_sobel, y_sobel, thresh=(0, 255)):
#     """Apply a threshold to the overall magnitude of the gradient
#
#     Args:
#         x_sobel (image): sobel image of gradients across the x axis
#
#         y_sobel (image): sobel image of gradients across the y axis
#
#         thresh (tuple): (lower_bound, upper_bound)
#                         lower_bound = lower bound for determining output
#                         upper_bound = upper bound for determining output
#
#     Returns:
#         image: binary threshold image
#     """
#     sobel = np.sqrt(x_sobel ** 2 + y_sobel ** 2)
#     scale_factor = np.max(sobel) / 255
#     sobel = (sobel / scale_factor).astype(np.uint8)
#     return binary_threshold(sobel, thresh)


# def dir_threshold(x_sobel, y_sobel, thresh=(0, np.pi / 4)):
#     """Apply direction threshold
#
#     Args:
#         x_sobel (image): sobel image of gradients across the x axis
#
#         y_sobel (image): sobel image of gradients across the y axis
#
#         thresh (tuple): (lower_bound, upper_bound)
#                         lower_bound = lower bound for determining output
#                         upper_bound = upper bound for determining output
#
#     Returns:
#         image: binary threshold image
#     """
#     sobel = np.arctan2(np.absolute(y_sobel), np.absolute(x_sobel))
#     return binary_threshold(sobel, thresh)


def combined_binary(img, ksize=3):
    """Apply gradien and colour binary threshold transform"""
    # ---------------------------------------------------------
    # Colour binary
    hls = cv2.cvtColor(np.copy(img), cv2.COLOR_RGB2HLS)
    r_binary = binary_threshold(img[:, :, 0], (225, 255))
    s_binary = binary_threshold(hls[:, :, 2], (170, 255))
    colour_binary = np.zeros_like(r_binary)
    colour_binary[((r_binary == 1) |
                   (s_binary == 1))] = 1

    # ---------------------------------------------------------
    # Sobel binary
    channel = hls[:, :, 2]
    x_sobel = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_binary = abs_sobel_thresh(x_sobel, thresh=(20, 100))

    # ---------------------------------------------------------
    # Combined binary
    combined_binary = np.zeros_like(r)
    combined_binary[((colour_binary == 1) |
                     (sobel_binary == 1))] = 1

    return combined_binary


def get_perspective_matrices():
    src_list = [(200, 720), (570, 470), (720, 470), (1130, 720)]
    dst_list = [(350, 720), (350, 0), (980, 0), (980, 720)]

    src = np.float32([src_list])
    dst = np.float32([dst_list])

    pers = cv2.getPerspectiveTransform(src, dst)
    pers_inv = cv2.getPerspectiveTransform(dst, src)

    return pers, pers_inv


def warper(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


def run_pipeline(img, M, dist):
    # locals
    pers, pers_inv = get_perspective_matrices()

    # pipeline
    img = undistort_img(img, M, dist)
    img = combined_binary(img)
    img = warper(img, pers)

    return img


if __name__ == "__main__":
    print('{} is to be imported not run directly'.format( __file__))
