import numpy as np
import torch


# Robust clipping bounds estimated from the Sen1Floods11
# hand-labeled Sentinel-1 dataset.
VV_MIN = -24.2971
VV_MAX = -2.3997

VH_MIN = -34.5019
VH_MAX = -9.5352


def preprocess_sentinel1(image):
    """
    Preprocess a Sentinel-1 image.

    Input:
        image: NumPy array with shape (2, H, W)
               Channel 0 = VV
               Channel 1 = VH

    Output:
        tensor: Float32 tensor with shape (2, H, W)
                Values approximately in [0, 1].

        valid_mask: Boolean tensor indicating pixels where
                    both Sentinel-1 channels contain valid data.
    """

    image = image.astype(np.float32).copy()

    vv = image[0]
    vh = image[1]

    # A pixel is valid only when both channels contain
    # finite values.
    valid_mask = np.isfinite(vv) & np.isfinite(vh)

    # Replace invalid values temporarily so clipping and
    # normalization cannot propagate NaNs.
    vv = np.nan_to_num(vv, nan=VV_MIN)
    vh = np.nan_to_num(vh, nan=VH_MIN)

    # Clip extreme outliers using the measured percentiles.
    vv = np.clip(vv, VV_MIN, VV_MAX)
    vh = np.clip(vh, VH_MIN, VH_MAX)

    # Scale independently to [0, 1].
    vv = (vv - VV_MIN) / (VV_MAX - VV_MIN)
    vh = (vh - VH_MIN) / (VH_MAX - VH_MIN)

    processed = np.stack([vv, vh], axis=0)

    tensor = torch.from_numpy(processed).float()
    valid_mask = torch.from_numpy(valid_mask)

    return tensor, valid_mask


def preprocess_label(label):
    """
    Convert Sen1Floods11 labels into a training-ready mask.

    Original labels:
        -1 = NoData
         0 = Not Water
         1 = Water

    Returns:
        label_tensor:
            0 = Not Water
            1 = Water
            -1 = Ignore / NoData
    """

    label = label.astype(np.int64).copy()

    label_tensor = torch.from_numpy(label)

    return label_tensor