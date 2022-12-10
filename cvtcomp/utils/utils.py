import pickle
from typing import Tuple, NoReturn
import os

import numpy as np
import cv2
import tqdm
from matplotlib import pyplot as plt, animation
from skimage.metrics import structural_similarity
from IPython.display import HTML
from cvtcomp.base import Encoder, Decoder, TensorVideo
from cvtcomp.io import load_video_to_numpy

def compute_ssim(video1, video2):

    ssim_val = structural_similarity(video1, video2, channel_axis=3)

    return ssim_val


def compute_metrics_dataset(folderpath: str, compression_type="tucker", chunk_size=30, quality=0.2) -> float:

    assert compression_type in ('tucker', 'tt'), "Compression type should be either 'tucker' or 'tt'"

    fnames = [fname for fname in os.listdir(folderpath) if fname[-4:] == ".y4m"]

    res_metrics_psnr = np.zeros(len(fnames))
    res_metrics_ssim = np.zeros(len(fnames))
    res_cr = None

    tensor_video = TensorVideo(
        compression_type=compression_type,
        quality=quality,
        chunk_size=chunk_size,
        decoded_data_type=np.uint8
    )

    total_size, compressed_size = 0.0, 0.0 # We calculate CR over the complete dataset

    for ii in tqdm.tqdm(range(len(fnames))):
        data, _, _, _ = load_video_to_numpy(os.path.join(folderpath, fnames[ii]))

        tensor_video.encode(data)
        restored_data = tensor_video.decode()

        compressed_size += tensor_video.encoded_data_size
        total_size += os.path.getsize(os.path.join(folderpath, fnames[ii]))

        res_metrics_psnr[ii] = cv2.PSNR(data, restored_data)
        res_metrics_ssim[ii] = compute_ssim(data, restored_data)

    res_cr = float(compressed_size) / float(total_size)

    return res_cr, np.mean(res_metrics_psnr), np.mean(res_metrics_ssim)


def delete_reference_frame(video):
    new_video = video.astype(np.int16) - video[0].astype(np.int16)
    return new_video, video[0]


def restore_from_reference_frame(video_without_ref_frame, ref_frame):
    video = video_without_ref_frame + ref_frame
    video[video > 255.0] = 255.0
    video[video < 0.0] = 0.0
    video = video.astype(np.uint8)
    return video


def play_video(video: np.array, fps: int = 30):
    """
    Plays video specified in np.array
    Input:
    :param: video - np.array [frames, height, width, channels] - RGB uint8
    :param: fps - fps :)
    """

    fig = plt.figure(figsize=(6,4))
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