import cv2
import glob
import matplotlib.pyplot as plt
import os.path

from calibrate_camera import find_corners, calibrate_camera
from dotenv import find_dotenv, load_dotenv
from helpers import save_image, watermark
from pipeline import undistort_img


def undistort_img_dir(in_img_dir, M, dist, out_img_dir=None):
    """Undistort test images and apply a watermark to the image if output
    directory to save to is provided.

    Args:
        in_img_dir (str): path to directory containing images to undistort

        M (numpy.array): camera matrix (output from cv2.calibrateCamera())

        dist (numpy.array): distortion coefficients (output from
                            cv2.calibrateCamera())

        out_img_dir (str): (OPTIONAL) if specified, undistorted images will be
                           saved to this directory.

    References:
        cv2.putText
         - http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#cv2.putText

        OpenCV fonts
         - https://codeyarns.com/2015/03/11/fonts-in-opencv/

        Transparent overlays with OpenCV
         - http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    """
    for path in glob.iglob(os.path.join(in_img_dir, '*.jpg')):
        img = cv2.imread(path)
        img = undistort_img(img, M, dist)

        # save and show image if requested
        if out_img_dir:
            # save locals
            file_name = os.path.split(path)[-1]
            img = watermark(img, 'UNDISTORTED')

            # save
            save_image(img, os.path.join(out_img_dir, file_name))


def undistort_chessboard(img_path, M, dist, out_img_dir=None):
    """Saves a before and after side-by-side comparison of a distorted image and
    an undistorted image.

    Args:
        in_img_dir (str): path to directory containing images to undistort

        M (numpy.array): camera matrix (output from cv2.calibrateCamera())

        dist (numpy.array): distortion coefficients (output from
                            cv2.calibrateCamera())
    """
    # set images
    distorted_image = cv2.imread(img_path)
    undistorted_img = undistort_img(distorted_image.copy(), M, dist)

    # build plot
    fig, axes = plt.subplots(1, 2, figsize=(5, 2))

    axes[0].imshow(distorted_image)
    axes[0].set_title('Distorted Image')

    axes[1].imshow(undistorted_img)
    axes[1].set_title('Undistorted Image')

    if out_img_dir:
        # save plot
        save_path = os.path.join(out_img_dir, 'undistort_chessboard.png')
        plt.savefig(save_path)
    else:
        plt.show()


def undistort_images():
    """Executed when script file is called directly"""
    # get env vars
    load_dotenv(find_dotenv())
    repo_dir = os.environ['REPO_DIR']

    # set working dirs
    cal_img_dir = os.path.join(repo_dir, 'camera_cal')
    dump_dir = os.path.join(repo_dir, 'data')
    out_img_dir = os.path.join(repo_dir, 'output_images')
    test_img_dir = os.path.join(repo_dir, 'test_images')

    # find corners
    obj_pts, img_pts = find_corners(cal_img_dir)

    # calibrate camera
    img_path = os.path.join(test_img_dir, 'test1.jpg')
    pickle_path = os.path.join(dump_dir, 'calibration.p')
    M, dist = calibrate_camera(obj_pts, img_pts, img_path, pickle_path)

    # undistort test images
    undistort_img_dir(test_img_dir, M, dist, out_img_dir)

    # undistort a chessboard image
    chessboard_path = os.path.join(out_img_dir, 'calibration3.jpg')
    undistort_chessboard(chessboard_path, M, dist, out_img_dir)


if __name__ == "__main__":
    # run if called directly
    undistort_images()
