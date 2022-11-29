import pickle

import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker, matrix_product_state
from tensorly import tt_to_tensor, tucker_to_tensor
import cv2


def tucker_encode(data: np.ndarray, quality=1.0):
    ranks = (
        max(1, int(data.shape[0] * quality)),
        int(data.shape[1] * quality),
        int(data.shape[2] * quality),
        3
    )

    compressed_data = tucker(
        tl.tensor(data.astype(np.float32)),
        rank=ranks,
        tol=1e-7,
        n_iter_max=300
    )

    return compressed_data


def tt_encode(data: np.ndarray, quality=1.0):
    ranks = (
        1,
        int(data.shape[0] * quality),
        int(data.shape[1] * quality),
        int(data.shape[2] * quality),
        1,
    )

    compressed_data = matrix_product_state(
        tl.tensor(data.astype(np.float32)),
        rank=ranks
    )

    return compressed_data


def tt_decode(compressed_data: list) -> np.ndarray:
    return tt_to_tensor(compressed_data)


def tucker_decode(compressed_data: list) -> np.ndarray:
    return tucker_to_tensor(compressed_data)


class Encoder:
    """Encoder, which adopts tensor decomposition approaches"""

    def __init__(self, **kwargs):
        """
        :param: quality in [0, 1] - rank ratio from the initial dimension along the t, w, h
        :param: encoder_type - type of the used tensor decomposition: 'tucker' or 'tt'""
        """
        self.quality = 1.0
        self.encoder_type = "tucker"
        self.encoder = tucker_encode

        for key, value in kwargs.items():
            if key == "encoder_type":
                self.encoder_type = value
                if self.encoder_type == "tucker":
                    self.encoder = tucker_encode
                elif self.encoder_type == "tt":
                    self.encoder = tt_encode
                else:
                    raise ValueError(f"Wrong encoder type: {self.encoder_type}")
            elif key == "quality":
                assert 0 <= value <= 1.0, "0 <= quality <= 1"
                self.quality = value
            else:
                raise ValueError(f"Wrong argument is provided : {value}")

    def encode(self, data: np.ndarray):
        return self.encoder(data, quality=self.quality)


class Decoder:
    """Decoder, which adopts tensor decomposition approaches"""

    def __init__(self, **kwargs):
        """
        :param: decoder_type - type of the used tensor decomposition: 'tucker' or 'tt'""
        """
        self.decoder_type = "tucker"
        self.decoder = tucker_decode

        for key, value in kwargs.items():
            if key == "decoder_type":
                self.decoder_type = value
                if self.decoder_type == "tucker":
                    self.decoder = tucker_decode
                elif self.decoder_type == "tt":
                    self.decoder = tt_decode
                else:
                    raise ValueError(f"Wrong decoder type: {self.decoder_type}")
            elif key == "quality":
                self.quality = value
            else:
                raise ValueError(f"Wrong argument is provided : {value}")

    def decode(self, compressed_data: list):

        decompressed_data = np.clip(self.decoder(compressed_data), 0.0, 255.0).astype(np.uint8)

        return decompressed_data


class StreamerEncoded:
    """Class for streaming the compressed data in to frame-wise manner"""

    def __init__(self, compressed_data, video_len, encoder_type="tucker"):
        self.compressed_data = compressed_data
        self.encoder_type = encoder_type
        self.n_frame = 0
        self.video_len = video_len

    def __next__(self):
        while self.n_frame < self.video_len:
            if self.encoder_type == 'tucker':
                raise NotImplementedError("TBD")
            elif self.encoder_type == 'tt':
                raise NotImplementedError("TBD")
            self.n_frame += 1
            yield None

    def __iter__(self):
        return self


class TensorVideo:
    def __init__(self, data=None, encoder_type='tt', quality=0.5, chunk_size=None, data_type=np.uint8):
        self.encoder_type = encoder_type
        self.quality = quality
        self.encoder = Encoder(encoder_type=self.encoder_type, quality=self.quality)
        self.decoder = Decoder(decoder_type=self.encoder_type)
        self.chunk_size = chunk_size
        self.shape = None
        self.data_type = data_type
        self.data = self.from_numpy(data) if data is not None else None

    def from_numpy(self, data: np.ndarray):
        res = []
        self.data_type = data.dtype
        if self.chunk_size is None:
            self.chunk_size = data.shape[0]
        self.shape = data.shape
        for frame_no in range(0, data.shape[0], self.chunk_size):
            chunk = data[frame_no:frame_no + self.chunk_size, :, :, :].astype(np.float64)
            coded_chunk = self.encoder.encode(chunk)
            res.append(coded_chunk)
        return res

    def to_numpy(self):
        res = []
        for coded_chunk in self.data:
            x = self.decoder.decode(coded_chunk)
            res.append(x)

        res = np.vstack(res)

        res[res > 255.0] = 255.0
        res[res < 0.0] = 0.0
        res = res.astype(self.data_type)
        return res

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.__dict__, file)

    def load(self, filename):
        with open(filename, 'rb') as file:
            self.__dict__ = pickle.load(file)
        return self

