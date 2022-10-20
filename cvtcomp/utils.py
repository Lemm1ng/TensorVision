from typing import Tuple, NoReturn
import os

import numpy as np
import cv2
import tqdm

from cvtcomp.base import *


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
    """
    Output
    :param: video <np.ndarray> - [F, W, H, C] uint8 RGB format
    :param: fourcc <int> - encoded codec type (!!!HARD TO FIND its specification) The website is owned by casino :)
    :param: fps <float> - fps...
    :param: size <(int, int)> - Width, Height
    :param: color <bool> - True if colored video should be saved
    """

    video_writer = cv2.VideoWriter(filepath, fourcc, fps, size, color)

    for ii in range(video.shape[0]):
        video_writer.write(cv2.cvtColor(video[ii, :, :, :], cv2.COLOR_RGB2BGR))

    video_writer.release()


def save_compressed_video(filepath: str, video: np.ndarray, metadata: dict) -> NoReturn:

    raise NotImplementedError("TBD")


def load_compressed_video(filepath: str) -> NoReturn:

    raise NotImplementedError("TBD")


def compute_metrics_dataset(folderpath: str, encoder: Encoder, decoder: Decoder, metric="psnr") -> float:

    if metric == "psnr":
        compute_metric = cv2.PSNR
    elif metric == "ssim":
        raise NotImplementedError
    else:
        raise ValueError(f"Wrong metric is specified: {metric}. Please , use 'psnr' or 'ssim'")

    assert encoder.encoder_type == decoder.decoder_type, "encoder and decoder should use the same compressed format"

    fnames = [fname for fname in os.listdir(folderpath) if fname[-4:] == ".y4m"]

    res_metrics = np.zeros(len(fnames))
    res_cr = np.zeros(len(fnames))

    for ii in tqdm.tqdm(range(len(fnames))):

        total_size, compressed_size = 0, 0

        data, _, _, _ = load_video_to_numpy(os.path.join(folderpath, fnames[ii]))

        total_size += data.nbytes
        compressed_data = encoder.encode(data)
        restored_data = decoder.decode(compressed_data)

        if encoder.encoder_type == 'tucker':
            compressed_size += compressed_data[0].nbytes
            compressed_size += sum([x.nbytes for x in compressed_data[1]])
        elif encoder.encoder_type == "tt":
            compressed_size += sum([x.nbytes for x in compressed_data])

        res_metrics[ii] = compute_metric(data, restored_data)
        res_cr[ii] = float(compressed_size) / float(total_size)

    return np.mean(res_metrics), np.mean(res_cr)


if __name__ == "__main__":
    pass