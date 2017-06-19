import cv2
import glob
import numpy as np
import os.path

from src.helpers import save_image, save_pickle


def find_corners(in_image_dir, out_img_dir=None):
    """Find corners on chessboard images to enable calibration of camera

    Args:
        in_image_dir (str): directory where chessboard images are found.

        out_img_dir (str): (Optional) directory to save the images that have
                                      annotated corners drawn on.

                                      Will also cause images to be shown on
                                      screen.

    Returns:
        tuple: (object_points, image_points)

        'object_points': list of inferred coordinates based on chessboard size.

        'image_points': list created as output from cv2.findChessboardCorners.
    """
    # set locals
    n_x = 6
    n_y = 9
    yx = (n_y, n_x)

    # calibrate if images directory is found
    if not os.path.isdir(in_image_dir):
        print("Calibration image path not found")
    else:
        # prepare object points for saving when corners are found
        obj_pts = np.zeros((n_y * n_x, 3), np.float32)
        obj_pts[:, :2] = np.mgrid[0:n_y, 0:n_x].T.reshape(-1, 2)

        # arrays for storing object points and corners
        object_points = []
        image_points = []

        # process each calibration image
        for path in glob.iglob(os.path.join(in_image_dir, '*.jpg')):
            file_name = os.path.split(path)[-1]
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # find chessboard corners in each image
            retval, corners = cv2.findChessboardCorners(gray, yx, None)
            print('{} corners found: {} '.format(file_name, retval))

            # add object points and image points if found
            if retval:
                object_points.append(obj_pts)
                image_points.append(corners)

                # draw corners on image, save and show image
                if out_img_dir:
                    cv2.drawChessboardCorners(img, yx, corners, retval)
                    save_image(img, os.path.join(out_img_dir, file_name))
                    cv2.imshow(file_name, img)
                    cv2.waitKey(500)

        # cleanup and write to stdout
        cv2.destroyAllWindows()
        print('object points: {}'.format(len(object_points)))
        print('image points: {}'.format(len(image_points)))

        return object_points, image_points


def calibrate_camera(object_points, image_points, img_path, pickle_path=None):
    """Calibrates the camera based on points

    Args:
        object_points (list): list of object points (numpy array).

        image_points (list): list of image points (numpy array).

        img_path (str): image to use for calibration.

        pickle_path (str): (OPTIONAL) path to dump calibration output to.

    Returns:
        tuple: (M, dist)

        'M': camera matrix output from cv2.calibrateCamera().

        'dist': distortion coefficients output from cv2.calibrateCamera().
    """
    if not os.path.isfile(img_path):
        print('Test image not found')
    else:
        # load image and get y (height) and x (width) dimensions
        # ignore channel dimension
        img = cv2.imread(img_path)
        img_size = img.shape[0:2]

        # calibrate
        _, M, dist, _, _ = cv2.calibrateCamera(
            object_points, image_points, img_size, None, None)

        # pickle output for later use if requested
        if pickle_path:
            calibration = {}
            calibration['M'] = M
            calibration['dist'] = dist
            save_pickle(calibration, pickle_path)

        return M, dist


if __name__ == "__main__":
    # not to be run directly
    pass
