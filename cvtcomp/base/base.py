import pickle

import numpy as np
from cvtcomp.base.decompositions import ttsvd_encode, tt_decode, tuckersvd_encode, tucker_decode


class Encoder:
    """Encoder, which adopts tensor decomposition approaches"""

    def __init__(self, **kwargs):
        """
        :param: quality PSNR < 45 db
        :param: encoder_type - type of the used tensor decomposition: 'tucker' or 'tt'""
        """
        self.quality = 25.0
        self.encoder_type = "tucker"
        self.encoder = tuckersvd_encode

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
                assert 0 <= value <= 45.0, "0.0 <= PSNR <= 45.0"
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


class TensorVideo:
    def __init__(self, compression_type='tt', quality=25.0, chunk_size=None, decoded_data_type=np.uint8):

        self.compression_type = compression_type
        self.quality = quality
        self.encoder = Encoder(encoder_type=self.compression_type, quality=self.quality)
        self.decoder = Decoder(decoder_type=self.compression_type)
        self.chunk_size = chunk_size
        self.decoded_data_type = decoded_data_type

        self.encoded_data = None
        self.shape = None
        self.encoded_data_size = None
        self.fps = None

    def encode(self, data: np.ndarray, show_results=False):

        self.encoded_data_size = 0
        res = []

        if self.chunk_size is None:
            self.chunk_size = data.shape[0]
        self.shape = data.shape
        for frame_no in range(0, data.shape[0], self.chunk_size):

            compressed_data_chunk = self.encoder.encode(data[frame_no:frame_no + self.chunk_size, :, :, :].astype(np.float32))

            if self.compression_type == 'tt':
                self.encoded_data_size += sum([x.nbytes for x in compressed_data_chunk])
            elif self.compression_type == 'tucker':
                self.encoded_data_size += compressed_data_chunk[0].nbytes
                self.encoded_data_size += sum([x.nbytes for x in compressed_data_chunk[1]])

            res.append(compressed_data_chunk)

        self.encoded_data = res

        if show_results:
            return self.encoded_data

    def decode(self):
        res = []
        for coded_chunk in self.encoded_data:
            x = self.decoder.decode(coded_chunk)
            res.append(x)

        res = np.vstack(res)

        res[res > 255.0] = 255.0
        res[res < 0.0] = 0.0
        res = res.astype(self.decoded_data_type)
        return res

    def save(self, filename, fps=30):
        self.fps=fps
        with open(filename, 'wb') as file:
            pickle.dump(self.__dict__, file)

    def load(self, filename):
        with open(filename, 'rb') as file:
            self.__dict__ = pickle.load(file)
        return self


    # NOT IMPLEMENTED YET
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
