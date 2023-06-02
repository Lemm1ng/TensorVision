from typing import List, Tuple

import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker, matrix_product_state
from tensorly import tt_to_tensor, tucker_to_tensor


def tucker_encode(data: np.ndarray, quality: float = 1.0) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Compress video in RGB24 format to the Tucker format. This implementation relies on tensorly package.
     Therefore, analytical selection of decomposition ranks is unavailable.
     Consider it a baseline for further improvements.

     Input:

     :param data  <np.nd.array>  video in RGB24 format

     :param quality  <float> rank / size along each dimension. Directly correlates with PSNR.
     However, the dependence is not linear.

     Output:

     compressed_data  <[Core, [Factors]]> - Compressed data in Tucker format.
     """

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
        n_iter_max=1
    )

    return compressed_data


def tt_encode(data: np.ndarray, quality: float = 1.0) -> List[np.ndarray]:
    """Compress video in RGB24 format to the Tensor-Train format. This implementation relies on tensorly package.
         Therefore, analytical selection of decomposition ranks is unavailable.
         Consider it a baseline for further improvements.

         Input:

         :param data  <np.nd.array>  video in RGB24 format.

         :param quality  <float> rank / size along each dimension. Directly correlates with PSNR.
         However, the dependence is not linear.

         Output:

         compressed_data  <[Factors[np.ndarray]]> - Compressed data in Tensor-Train format.
         """

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


def tt_decode(compressed_data: List[np.ndarray]) -> np.ndarray:
    """Decode video in Tensor-Train format to the RGB24 format. All data is in np.uint8 format

    Input:

    :param compressed_data - Compressed data in Tensor-Train format.

    Output:

    compressed_data  <[Factors[np.ndarray]]> - Compressed data in Tensor-Train format.
    """

    return tt_to_tensor(compressed_data)


def tucker_decode(compressed_data: Tuple[np.ndarray, List[np.ndarray]]) -> np.ndarray:
    """Decode video in Tucker format to the RGB24 format. All data is in np.uint8 format

        Input:

        :param compressed_data - Compressed data in Tensor-Train format.

        Output:

        compressed_data  <[Factors[np.ndarray]]> - Compressed data in Tensor-Train format.
        """

    return tucker_to_tensor(compressed_data)


# Custom implementation of tensor decompositions
def unfold(tensor: np.ndarray[np.float32], mode: int) -> np.ndarray[np.float32]:
    """Unfolds tensor along the specified axis.

    Input:

    :param tensor <np.ndarray> input tensor (e.g. video).
    :param mode  <int> - axis to unfold

    Output:

    unfolded_tensor  <np.ndarray>  - matrix, representing the unfolded tensor with first axis specified in mode.
    """

    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


def truncated_svd(matrix: np.ndarray[np.float32], quality: float, verbose=False) -> Tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
    """Performs truncated svd decomposition with analytical rank selection.
     It is based on desired quality of compression. Frobenius norm is used.

     Input:

     :param matrix  <np.ndarray> - input matrix
     :param quality  <float>  ||matrix - matrix_r||_F <= quality
     Here, ||*||_F is Frobenius norm, matrix_r - best rank_r approximation of matrix.

     Output:
    U_r, V_r - (np.ndarray, np.ndarray, int) Skeleton decomposition based on truncated SVD
    and its rank(for convenience).
     """

    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    res_r_frob = 0.0  # squared Frobenius norm of residuals
    r = 1  # Minimal rank for decomposition
    for ii in range(s.size - 1, -1, -1):
        res_r_frob += s[ii] ** 2
        if np.sqrt(res_r_frob) >= quality:
            r = ii + 1
            break
    if verbose:
        return u[:, :r], (s[:r] * vh[:r, :].T).T, s
    else:
        return u[:, :r], (s[:r] * vh[:r, :].T).T


def fold(unfolded_tensor: np.ndarray[np.float32], mode: int, shape: Tuple[int]) -> np.ndarray[np.float32]:
    """Folds the matrix representation of tensor along to the tensor of specified shape.

    Input:

    :param unfolded_tensor  <np.ndarray> - input matrix
    :param mode  <int> - mode which was used to unfold original tensor.
    :param shape <Tuple[int]> - shape of original tensor.

    Output:

    tensor  <np.ndarray> - folded tensor
    """

    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)

    return np.moveaxis(np.reshape(unfolded_tensor, full_shape), 0, mode)


def ttsvd_encode(tensor: np.ndarray[np.uint8], quality: float = 25.0, video_range: float = 255, heuristics: bool = True, verbose:bool = False) -> Tuple[np.ndarray[np.float32]]:
    """Tensor-Train decomposition with analytical rank computation based on PSNR and its connection with Frobenius norm.
    https://github.com/azamat11235/NLRTA - The basic algorithm implementation is adopted from this source.

    Input:

    :param tensor   <np.ndarray[np.uint8]> - input tensor (e.g. video)
    :param quality  <float> - Target PSNR (db) for tensor compression. Please, avoid using quality > 45 db
    since the numerical errors do not allow reaching it.

    :param video_range <int> - max signal value in tensor

    :param verbose <bool> - return singular values for the trSVD

    Output:

    compressed_data  <Tuple[np.ndarray[np.float32]]> - compressed data in Tensor-Train format
    """

    assert quality <= 60.0, "Please, use quality <= 60.0 db"

    if verbose:
        sigma_all = []

    tensor = tensor.astype(np.float32)
    # Maximum allowed Frobenius norm for residuals based on target PSNR
    if heuristics:
        # Whe do not compress along the small dims (e.g. C)
        r_min_to_compress = 4

        single_svd_decomp_quality = np.sqrt(
            (tensor.size * video_range ** 2 / np.power(10, quality / 10))
            / (sum([x >= r_min_to_compress for x in tensor.shape]) - 1)
        )
    else:
        single_svd_decomp_quality = np.sqrt(
                (tensor.size * video_range**2 / np.power(10, quality / 10))
                / (len(tensor.shape) - 1))

    n = np.array(tensor.shape)
    G_list = []
    G = tensor.copy()
    G0 = unfold(G, 0)

    if verbose:
        if heuristics:
            if min(G0.shape) < r_min_to_compress:
                u, vh, sigma = truncated_svd(G0, 0, verbose=verbose)
            else:
                u, vh, sigma = truncated_svd(G0, single_svd_decomp_quality, verbose=verbose)
        else:
            u, vh, sigma = truncated_svd(G0, single_svd_decomp_quality, verbose=verbose)
        sigma_all.append(sigma)
    else:
        if heuristics:
            if min(G0.shape) < r_min_to_compress:
                u, vh = truncated_svd(G0, 0, verbose=verbose)
            else:
                u, vh = truncated_svd(G0, single_svd_decomp_quality, verbose=verbose)
        else:
            u, vh = truncated_svd(G0, single_svd_decomp_quality, verbose=verbose)

    r_prev = u.shape[1]
    G_list.append(u)
    for k in range(1, len(tensor.shape) - 1):
        vh = vh.reshape(r_prev * n[k], np.prod(n[k + 1:]))
        if verbose:
            if heuristics:
                if min(vh.shape) < r_min_to_compress:
                    u, vh, sigma = truncated_svd(vh, 0, verbose=verbose)
                else:
                    u, vh, sigma = truncated_svd(vh, single_svd_decomp_quality, verbose=verbose)
            sigma_all.append(sigma)
        else:
            if heuristics:
                if min(vh.shape) < r_min_to_compress:
                    u, vh = truncated_svd(vh, 0, verbose=verbose)
                else:
                    u, vh = truncated_svd(vh, single_svd_decomp_quality, verbose=verbose)
        r = u.shape[1]
        r_cur = min(r, vh.shape[0])
        u = u.reshape(r_prev, n[k], r_cur)
        G_list.append(u)
        r_prev = r

    G_list.append(vh)
    G_list[0] = np.expand_dims(G_list[0], 0)
    G_list[-1] = np.expand_dims(G_list[-1], -1)
    if verbose:
        return G_list, sigma_all
    else:
        return G_list


def tuckersvd_encode(tensor: np.ndarray[np.uint8], quality: float = 25.0, video_range: float = 255, heuristics: bool = True, verbose=False) -> Tuple[np.ndarray[np.float32], List[np.ndarray[np.float32]]]:
    """Tucker decomposition with analytical rank computation based on PSNR and its connection with Frobenius norm.
        https://github.com/azamat11235/NLRTA - The basic algorithm implementation is adopted from this source.

        Input:

        :param tensor  <np.ndarray[np.uint8]> - input tensor (e.g. video)
        :param quality  <float> - Target PSNR (db) for tensor compression. Please, avoid using quality > 45 db
        since the numerical errors do not allow reaching it.
        :param heuristics <bool> - if True enables the heuristics in analytical rank evaluation during the compression.

        :param video_range <int> - max signal value in tensor
        :param verbose <bool> - return singular values for the trSVD

        Output:

        compressed_data  <Tuple[np.ndarray[np.float32], List[np.ndarray[np.float32]]> - compressed data in Tensor-Train format
        """

    S = tensor.astype(np.float32)

    if verbose:
        sigma_all = []

    # Maximum allowed Frobenius norm for residuals based on target PSNR
    if heuristics:
        # Whe do not compress along the small dims (e.g. C)
        r_min_to_compress = 4

        single_svd_decomp_quality = np.sqrt(
            (tensor.size * video_range ** 2 / np.power(10, quality / 10))
            / sum([x >= r_min_to_compress for x in tensor.shape])
        )
    else:
        # No heuristics:
        single_svd_decomp_quality = np.sqrt(
            (tensor.size * video_range ** 2 / np.power(10, quality / 10))
            / len(tensor.shape)
        )

    U_list = []
    for k in range(len(tensor.shape)):
        ak = unfold(S, k)

        if heuristics:
            if tensor.shape[k] < r_min_to_compress:
                if verbose:
                    u, vh, sigma = truncated_svd(ak, quality=0, verbose=verbose)
                    sigma_all.append(sigma)
                else:
                    u, vh = truncated_svd(ak, quality=0, verbose=verbose)
            else:
                if verbose:
                    u, vh, sigma = truncated_svd(ak, quality=single_svd_decomp_quality, verbose=verbose)
                    sigma_all.append(sigma)
                else:
                    u, vh = truncated_svd(ak, quality=single_svd_decomp_quality, verbose=verbose)
        else:
            if verbose:
                u, vh, sigma = truncated_svd(ak, quality=single_svd_decomp_quality, verbose=verbose)
                sigma_all.append(sigma)
            else:
                u, vh = truncated_svd(ak, quality=single_svd_decomp_quality, verbose=verbose)

        r = u.shape[1]
        shape = list(S.shape)
        shape[k] = min(vh.shape[1], r)
        S = fold(vh, k, shape)
        U_list.append(u)

    if verbose:
        return (S, U_list), sigma_all
    else:
        return S, U_list


if __name__ == "__main__":
    psnr = 45
    tst = 100 * np.ones((40, 32, 10, 3))
    tst[:, :, 0, 0] += 100 * np.ones((40, 32))
    print([x.shape for x in tuckersvd_encode(tst, psnr)[1]])
    print(np.linalg.norm(tst - tucker_decode(tuckersvd_encode(tst, psnr))))
