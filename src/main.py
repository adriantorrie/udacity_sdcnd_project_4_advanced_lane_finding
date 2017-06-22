import argparse
import cv2
import os.path
import pickle

from pipeline import Lane, get_perspective_matrices, run_pipeline


def main(args):
    # parse args
    source_path = args.s
    calibration = pickle.load( open(args.c, 'rb'))
    output_path = args.o
    img_width_x = args.w
    img_height_y = args.y

    # check file exists before processing
    if os.path.isfile(source_path):
        # locals
        M = calibration['M']
        dist = calibration['dist']
        left_lane = Lane()
        right_lane = Lane()
        pers, pers_inv = get_perspective_matrices()
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, codec, 20.0,
                              (img_height_y, img_width_x))

        # process video
        cap = cv2.VideoCapture(source_path)
        while (cap.isOpened()):
            ret, frame = cap.read()

            # new frame found
            if ret == True:
                # annotate frame and write to file
                frame = pipeline(frame, M, dist, pers, pers_inv, left_lane,
                                 right_lane)
                out.write(frame)

                # display annotated frame to screen
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    else:
        print('Source file not found')


if __name__ == "__main__":
    # set args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', help='path to source file', required=True)
    parser.add_argument('-c', help='path to pickled calibration', required=True)
    parser.add_argument('-o', help='path to output file', required=True)
    parser.add_argument('-y', help='height of video', required=True)
    parser.add_argument('-x', help='width of video', required=True)

    # run
    main(parser.parse_args())
