import cv2


def undistort_img(img_path, M, dist):
    """Un-distorts image on disk

    Args:
        img_path (str): Path to image to undistort

        M (numpy.array): Camera matrix (output from cv2.calibrateCamera())

        dist (numpy.array): Distortion coefficients (output from
                            cv2.calibrateCamera())

    Returns
        image: numpy.array representation of an image

    """
    return cv2.undistort(cv2.imread(img_path), M, dist, None, M)
