import pickle
from typing import List, Tuple, Union, NoReturn

import numpy as np
from cvtcomp.base.decompositions import ttsvd_encode, tt_decode, tuckersvd_encode, tucker_decode


class Encoder:
    """Encoder, which adopts tensor decomposition approaches including the stHOSVD and TTSVD"""

    def __init__(self, **kwargs):
        """
        :param quality PSNR < 45 db
        :param encoder_type - type of the used tensor decomposition: 'tucker' or 'tt'.
        'tucker' = stHOSVD, 'tt' = TTSVD
        :param verbose - provide additional data if needed
        """

        self.quality = 25.0
        self.encoder_type = "tucker"
        self.encoder = tuckersvd_encode
        self.verbose = False

        for key, value in kwargs.items():
            if key == "encoder_type":
                self.encoder_type = value
                if self.encoder_type == "tucker":
                    self.encoder = tuckersvd_encode
                elif self.encoder_type == "tt":
                    self.encoder = ttsvd_encode
                else:
                    raise ValueError(f"Wrong encoder type: {self.encoder_type}")
            elif key == "quality":
                assert 0 <= value <= 60.0, "0.0 <= PSNR <= 60.0"
                self.quality = value
            elif key == "verbose":
                self.verbose = value
            else:
                raise ValueError(f"Wrong argument is provided : {value}")

    def encode(self, data: np.ndarray[np.uint8]) -> Union[Tuple[np.ndarray, List[np.ndarray]], List[np.ndarray]]:
        """Encode the raw video
        :param data - raw video in RGB24 format (np.uint8)"""

        return self.encoder(data, quality=self.quality, verbose=self.verbose)


class Decoder:
    """Decoder, which adopts tensor decomposition approaches"""

    def __init__(self, **kwargs):
        """
        :param decoder_type - type of the used tensor decomposition: 'tucker' or 'tt'
        'tucker' = stHOSVD, 'tt' = TTSVD
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
            else:
                raise ValueError(f"Wrong argument is provided : {value}")

    def decode(self, compressed_data: Union[Tuple[np.ndarray, List[np.ndarray]], List[np.ndarray]]) -> np.ndarray[np.uint8]:
        """Decode the video to RGB24 format
        :param compressed_data:
        """

        decompressed_data = np.clip(self.decoder(compressed_data), 0.0, 255.0).astype(np.uint8)

        return decompressed_data


class TensorVideo:
    """Class, which provides basic functionality:
     1) Video encoding/decoding using the stHOSVD or TTSVD algorithms
     2) Save/load video in the encoded(compressed) format using the pickle package.
    """

    def __init__(self, compression_type='tt', quality=25.0, chunk_size=None, decoded_data_type=np.uint8, verbose=False):
        """
        :param compression_type - type of the used tensor decomposition ('tucker' or 'tt')
        'tucker' = stHOSVD, 'tt' = TTSVD
        :param quality - PSNR, dB. Should be less than  45 dB
        :param chunk_size - number of frames in the chunks video is divided by
        :param decoded_data_type - np.uint for RGB24
        :param verbose - provide additional data if needed
        """

        self.compression_type = compression_type
        self.quality = quality
        self.verbose = verbose
        self.encoder = Encoder(encoder_type=self.compression_type, quality=self.quality, verbose=self.verbose)
        self.decoder = Decoder(decoder_type=self.compression_type)
        self.chunk_size = chunk_size
        self.decoded_data_type = decoded_data_type

        self.encoded_data = None
        self.shape = None
        self.encoded_data_size = None
        self.fps = None

        self.metadata = []

    def encode(self, data: np.ndarray[np.uint8]) -> NoReturn:
        """Encode the raw video
        :param data - raw video in RGB24 format (np.uint8)"""

        self.encoded_data_size = 0
        res = []

        if self.chunk_size is None:
            self.chunk_size = data.shape[0]
        self.shape = data.shape
        for frame_no in range(0, data.shape[0], self.chunk_size):
            if self.verbose:
                compressed_data_chunk, metadata = self.encoder.encode(
                    data[frame_no:frame_no + self.chunk_size, ...].astype(np.float32)
                )
                self.metadata.append(metadata)
            else:
                compressed_data_chunk = self.encoder.encode(
                    data[frame_no:frame_no + self.chunk_size, ...].astype(np.float32)
                )

            if self.compression_type == 'tt':
                self.encoded_data_size += sum([x.nbytes for x in compressed_data_chunk])
            elif self.compression_type == 'tucker':
                self.encoded_data_size += compressed_data_chunk[0].nbytes
                self.encoded_data_size += sum([x.nbytes for x in compressed_data_chunk[1]])

            res.append(compressed_data_chunk)

        self.encoded_data = res

    def decode(self) -> np.ndarray:
        """Decode the video to RGB24 format
        """

        res = []
        for coded_chunk in self.encoded_data:
            x = self.decoder.decode(coded_chunk)
            res.append(x)

        res = np.vstack(res)

        res[res > 255.0] = 255.0
        res[res < 0.0] = 0.0
        res = res.astype(self.decoded_data_type)

        return res

    def save(self, filename: str, fps: int = 30) -> NoReturn:
        """
        Saves the encoded video
        :param filename - filepath to save encoded video
        :param fps - frames per second
        """
        self.fps = fps
        with open(filename, 'wb') as file:
            pickle.dump(self.__dict__, file)

    def load(self, filename: str):
        with open(filename, 'rb') as file:
            self.__dict__ = pickle.load(file)
        return self


class StreamerEncoded:
    """Class for streaming the compressed data in to frame-wise manner"""

    def __init__(self, compressed_data, video_len, encoder_type="tucker"):

        self.compressed_data = compressed_data
        self.encoder_type = encoder_type
        self.n_frame = 0
        self.video_len = video_len
        raise NotImplementedError("TBD")

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
