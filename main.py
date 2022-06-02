"""
Lane Lines Detection pipeline

Usage:
    main.py [--video] INPUT_PATH OUTPUT_PATH 

Options:

-h --help                               show this screen
--video                                 process video file instead of image
"""


import numpy as np
import matplotlib.image as mpimg
import cv2
import pyrealsense2 as rs
import docopt
from moviepy.editor import VideoFileClip
from lanelines import*


class FindLaneLines:

    def get_stream(stream):
        if stream == "color":
            pipeline = rs.pipeline()

            config = rs.config()
            config.enable_stream(rs.stream.color, 1280,
                                 720, rs.format.bgr8, 30)

            pipeline.start.config

            try:
                # Get Frameset of Color Sensor
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                img = np.asanyarray(color_frame.get_date())
                return img
            except:
                print("Color_frame konnte nicht gefunden werden")

    def show_result(img):
        while True:
            cv2.imshow('Result', img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    def process_cam():
        proc_img = LaneLines.forward(FindLaneLines.get_stream("color"))
        FindLaneLines.show_result(proc_img)


def main():
    FindLaneLines.process_cam()


if __name__ == "__main__":

    main()
