import cv2
import pickle


def save_image(img, file_path):
    """Saves image using cv2.imwrite with stdout confirmation of save path

    Args:
        img (numpy.array): image to save.

        file_path (str): full path to save image to.
    """
    cv2.imwrite(file_path, img)
    print('image saved: {}'.format(file_path))


def save_pickle(obj, pickle_path):
    """Saves object as pickled file with stdout confirmation of save path

    Args:
        obj (object): object to save.

        pickle_path (str): full path to save image to.
    """
    with open(pickle_path, 'wb') as p:
        pickle.dump(obj, p)
    print('calibration dumped: {}'.format(pickle_path))


def watermark(img, s):
    """Watermarks an image with specified string

    Args:
        img (numpy.array): array representation of an image

        s (str): the string to use as the watermark

    Returns:
        img: watermarked image (red font, with alpha = 0.3)
    """
    # locals
    orig = (img.shape[0] // 2, img.shape[1] // 3)
    font = cv2.FONT_HERSHEY_DUPLEX
    fsize = 3
    color = (211, 211, 211)
    thickness = 4
    alpha = 0.3

    # create watermark
    watermark = img.copy()
    watermark = cv2.putText(watermark, s, orig, font, fsize, color, thickness)
    cv2.addWeighted(watermark, alpha, img, 1 - alpha, 0, img)

    return watermark

if __name__ == "__main__":
    # not to be run directly
    pass
