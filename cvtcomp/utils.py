import pickle
from typing import Tuple, NoReturn
import os

import numpy as np
import cv2
import tqdm
from matplotlib import pyplot as plt, animation
from skimage.metrics import structural_similarity
from IPython.display import HTML
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


def save_compressed_video(filepath: str, compressed_video: list) -> NoReturn:

    with open(filepath, 'wb') as file:
        pickle.dump(compressed_video, file)


def load_compressed_video(filepath: str):

    with open(filepath, 'rb') as file:
        compressed_video = pickle.load(file)

    return compressed_video


def compute_ssim(video1, video2):

    ssim_val = structural_similarity(video1, video2, channel_axis=3)

    return ssim_val


def encdec_video_chunkwise(video, encoder, decoder, chunk_size=10):

    decompressed_video = np.zeros(video.shape, dtype=np.uint8)

    for ii in range(0, video.shape[0], chunk_size):
        compressed_chunk = encoder.encode(video[ii: ii + chunk_size, :, :, :])
        decompressed_video[ii: ii + chunk_size, :, :, :] = decoder.decode(compressed_chunk)

    return decompressed_video


def compute_metrics_dataset(folderpath: str, encoder: Encoder, decoder: Decoder, metric="psnr") -> float:
    if metric == "psnr":
        compute_metric = cv2.PSNR
    elif metric == "ssim":
        compute_metric = compute_ssim
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


def plot_ssim_ms_ssim(filepath: str, encoder_type='tucker', qualities=(0.10, 0.25, 0.50)):
    result_ssim = []
    video, *_ = load_video_to_numpy(filepath)

    for quality in qualities:
        decompressed_video = TensorVideo(
            data=video,
            encoder_type=encoder_type,
            quality=quality,
            chunk_size=50
        ).to_numpy()

        ssim_val = structural_similarity(video, decompressed_video, channel_axis=3)
        result_ssim.append(ssim_val)

    result_ssim = np.array(result_ssim)
    x = np.array(qualities)

    plt.figure(figsize=(8, 6))
    plt.grid(True)
    plt.xlabel("Core size")
    plt.ylabel("ms_ssim, ssim")
    plt.plot(x, result_ssim)


def convert_from_y4m_to_avi(filename, encoder_type='tucker', quality=1.0, chunk_size=50):
    video, fourcc, fps, size = load_video_to_numpy(filename)
    decompressed_video = TensorVideo(
        video,
        encoder_type=encoder_type,
        quality=quality,
        chunk_size=chunk_size
    ).to_numpy()
    save_video_from_numpy(filename[:-4] + '.avi', decompressed_video, fourcc, fps, size, color=True)


def delete_reference_frame(video):
    new_video = video.astype(np.int16) - video[0].astype(np.int16)
    return new_video[1:], video[0]


def restore_from_reference_frame(video_without_ref_frame, ref_frame):
    video = video_without_ref_frame + ref_frame
    video = np.vstack([ref_frame.reshape((1, ref_frame.shape[0], ref_frame.shape[1], ref_frame.shape[2])), video])
    video[video > 255.0] = 255.0
    video[video < 0.0] = 0.0
    video = video.astype(np.uint8)
    return video


def bin_search_video_metric(video, target_metric_value, num_iter=5, metric='psnr', encoder_type='tt'):
    if metric == "psnr":
        compute_metric = cv2.PSNR
    elif metric == "ssim":
        compute_metric = compute_ssim
    else:
        raise ValueError(f"Wrong metric is specified: {metric}. Please , use 'psnr' or 'ssim'")

    min_value = 0.0
    max_value = 1.0
    value, metric_value, compressed_video = 0, 0, None
    for _ in range(num_iter):
        value = (max_value + min_value) / 2
        compressed_video = TensorVideo(video, encoder_type=encoder_type, quality=value, chunk_size=50).to_numpy()
        metric_value = compute_metric(video, compressed_video)
        print(value, metric_value)
        if metric_value > target_metric_value:
            max_value = value
        else:
            min_value = value

    return value, metric_value, compressed_video


def play_video(video: np.array, fps: int = 30):
    """
    Plays video specified in np.array
    Input:
    :param: video - np.array [frames, height, width, channels] - RGB uint8
    :param: fps - fps :)
    """
    fig = plt.figure(figsize=(12,6))
    im = plt.imshow(video[0,:,:,:])
    plt.close() # this is required to not display the generated image

    def init():
        im.set_data(video[0,:,:,:])

    def animate(i):
        im.set_data(video[i,:,:,:])
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=video.shape[0],
                                   interval=1000.0 / fps)
    return HTML(anim.to_html5_video())


if __name__ == "__main__":
    pass