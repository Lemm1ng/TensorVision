from typing import Tuple, NoReturn
import pickle

import cv2
import numpy as np


def load_video_to_numpy(filepath: str) -> Tuple[np.ndarray, int, int, Tuple[int, int]]:
    """
    Input
    :param: filepath <str> - Filepath to the video to load
    Output
    video <np.ndarray> - [F, W, H, C] uint8
    fourcc <int> - encoded codec type (!!!MAGICK, HARD TO FIND its specification) The website is owned by casino :)
    fps <float> - fps...
    size <(int, int)> - Width, Height
    """

    video_capture = cv2.VideoCapture(filepath, cv2.CAP_FFMPEG)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    fourcc = int(video_capture.get(cv2.CAP_PROP_FOURCC))
    size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    video = []
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video.append(frame)
        else:
            video_capture.release()

    video = np.array(video)
    video_capture.release()

    return video, fourcc, fps, size


def save_video_from_numpy(filepath: str, video: np.array, fourcc: int, fps: int, size, color=True) -> NoReturn:
    """ Saves video in numpy RGB24 to YUV420. Please specify output file as .avi

    Input
    :param: filepath <str> - file to save video
    :param: video <np.ndarray> - [F, W, H, C] uint8 RGB format
    :param: fourcc <int> - encoded codec type (!!!HARD TO FIND its specification) The website is owned by casino now :)
    :param: fps <float> - fps...
    :param: size <(int, int)> - Width, Height
    :param: color <bool> - True if colored video should be saved
    """

    video_writer = cv2.VideoWriter(filepath, fourcc, fps, size, color)

    for ii in range(video.shape[0]):
        video_writer.write(cv2.cvtColor(video[ii, :, :, :], cv2.COLOR_RGB2BGR))

    video_writer.release()


def save_compressed_video(filepath: str, compressed_video: list) -> NoReturn:
    """Saves raw compressed video using the pickle package"""
    with open(filepath, 'wb') as file:
        pickle.dump(compressed_video, file)


def load_compressed_video(filepath: str):
    """Loads raw compressed video using the pickle package"""
    with open(filepath, 'rb') as file:
        compressed_video = pickle.load(file)

    return compressed_video
