import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker, matrix_product_state
from tensorly import tt_to_tensor, tucker_to_tensor
import cv2


def tucker_encode(data: np.ndarray, quality=1.0):

    ranks = (
        int(data.shape[0] * quality),
        int(data.shape[1] * quality),
        int(data.shape[2] * quality),
        3
    )

    compressed_data = tucker(
        tl.tensor(data.astype(np.float32)),
        rank=ranks
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
                assert value <= 1.0, "quality <= 1"
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

        decompressed_data = self.decoder(compressed_data).astype(np.uint8)
        decompressed_data[decompressed_data < 0] = 0
        decompressed_data[decompressed_data > 255] = 255

        return decompressed_data


    class StreamerEncoded:
        """Class for streamming the compressed data in a frame-wise manner"""

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
