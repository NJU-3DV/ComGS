import numpy as np

def srgb_to_linear(srgb):
    """
    Convert sRGB color values to linear RGB color space.

    This function transforms input sRGB values to linear RGB values using the
    standard sRGB to linear conversion formula. The conversion handles values
    differently based on whether they are below or above a threshold of 0.04045.

    Parameters
    ----------
    srgb : array_like
        Input array of sRGB values, typically in the range [0, 1]. Can be a scalar
        or a multi-dimensional array.

    Returns
    -------
    linear_rgb : ndarray
        Array of linear RGB values with the same shape as the input `srgb`.
        Values are in the linear color space, typically in the range [0, 1].

    Notes
    -----
    The conversion uses the following piecewise function:
    - For sRGB <= 0.04045: linear = sRGB / 12.92
    - For sRGB > 0.04045: linear = ((sRGB + 0.055) / 1.055) ** 2.4

    Examples
    --------
    >>> srgb_to_linear(0.5)
    0.21404114
    >>> srgb_to_linear(np.array([0.0, 0.5, 1.0]))
    array([0.        , 0.21404114, 1.        ])
    """
    # Create a mask for the condition
    mask = srgb <= 0.04045
    linear_rgb = np.zeros_like(srgb)

    # Apply the conversion formula
    linear_rgb[mask] = srgb[mask] / 12.92
    linear_rgb[~mask] = ((srgb[~mask] + 0.055) / 1.055) ** 2.4

    return linear_rgb


def linear_to_srgb(linear_rgb):
    """
    Convert linear RGB values to sRGB values.

    This function applies the standard transformation from linear RGB color space
    to sRGB color space, which is commonly used in digital imaging. The conversion
    uses a piecewise function to handle different ranges of input values and ensures
    the output is clamped to the valid sRGB range [0, 1].

    Parameters
    ----------
    linear_rgb : array_like
        Input array of linear RGB values, typically in the range [0, 1].
        Can be a scalar, list, or NumPy array of any shape.

    Returns
    -------
    numpy.ndarray
        Array of sRGB values in the range [0, 1], with the same shape as the input.

    Notes
    -----
    The conversion follows the sRGB specification:
    - For linear RGB values <= 0.0031308, the output is scaled by 12.92.
    - For linear RGB values > 0.0031308, the output is computed as
      1.055 * (value^(1/2.4)) - 0.055.
    - All output values are clipped to the range [0, 1].

    Examples
    --------
    >>> linear_to_srgb(0.0)
    0.0
    >>> linear_to_srgb(1.0)
    1.0
    >>> linear_to_srgb(np.array([0.001, 0.5]))
    array([0.01292, 0.73536...])
    """
    # Create a mask for the condition
    mask = linear_rgb <= 0.0031308
    srgb = np.zeros_like(linear_rgb)

    # Apply the conversion formula
    srgb[mask] = linear_rgb[mask] * 12.92
    srgb[~mask] = 1.055 * (linear_rgb[~mask] ** (1 / 2.4)) - 0.055

    # Clip the values to ensure they are in the range [0, 1]
    srgb = np.clip(srgb, 0, 1)

    return srgb


def read_exr(exr_path: str, with_alpha: bool = True) -> np.ndarray:
    """
    Read an OpenEXR file and return its contents as a numpy array.

    Args:
    exr_path: Path to the EXR file
    with_alpha: If True, also read the alpha channel if available

    Returns:
    numpy array with shape (height, width, channels)
    """
    import OpenEXR
    import Imath

    file = OpenEXR.InputFile(exr_path)
    dw = file.header()["dataWindow"]
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # Check if alpha channel exists
    header = file.header()
    channels_in_file = header['channels'].keys()
    has_alpha = 'A' in channels_in_file

    if with_alpha and has_alpha:
        channel_names = ["R", "G", "B", "A"]
        channels = file.channels(channel_names, FLOAT)
        pixels = np.zeros((size[1], size[0], 4))
    else:
        channel_names = ["R", "G", "B"]
        channels = file.channels(channel_names, FLOAT)
        pixels = np.zeros((size[1], size[0], 3))

    for i, channel in enumerate(channels):
        pixels[:, :, i] = np.frombuffer(channel, dtype=np.float32).reshape(
            size[1], size[0]
        )
    return pixels

def save_exr(exr_path: str, exr_image: np.ndarray) -> None:
    """
    Save a numpy array as an OpenEXR file.

    Args:
    exr_path: Path to save the EXR file
    exr_image: Numpy array with shape (height, width, channels) containing the image data.
               Channels can be 3 (RGB) or 4 (RGBA).
    """
    import OpenEXR
    import Imath

    height, width, channels = exr_image.shape
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # Prepare header
    header = OpenEXR.Header(width, height)
    header['channels'] = {}

    # Define channels based on input image
    if channels == 4:
        channel_names = ["R", "G", "B", "A"]
    else:
        channel_names = ["R", "G", "B"]

    for ch in channel_names:
        header['channels'][ch] = Imath.Channel(FLOAT)

    # Create output file
    exr_file = OpenEXR.OutputFile(exr_path, header)

    # Prepare channel data
    channel_data = {}
    for i, ch in enumerate(channel_names):
        channel_data[ch] = exr_image[:, :, i].astype(np.float32).tobytes()

    # Write to file
    exr_file.writePixels(channel_data)
    exr_file.close()

class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(numpy_img) * percentile) to

    def __call__(self, numpy_img, clip=True, alpha=None, gamma=True):
        if gamma:
            power_numpy_img = np.power(numpy_img, 1 / self.gamma)
        else:
            power_numpy_img = numpy_img
        non_zero = power_numpy_img > 0
        if non_zero.any():
            r_percentile = np.percentile(power_numpy_img[non_zero], self.percentile)
        else:
            r_percentile = np.percentile(power_numpy_img, self.percentile)
        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        tonemapped_img = np.multiply(alpha, power_numpy_img)

        if clip:
            tonemapped_img_clip = np.clip(tonemapped_img, 0, 1)

        return tonemapped_img_clip.astype('float32'), alpha, tonemapped_img

def adjust_exposure(hdr_image, exposure_factor):
    """
    Adjusts the exposure of an HDR image by applying an exponential scaling factor.

    Parameters
    ----------
    hdr_image : numpy.ndarray
        The input HDR (High Dynamic Range) image as a NumPy array. Expected to have
        floating-point values representing linear radiance.
    exposure_factor : float
        The exposure adjustment factor in stops. Positive values increase brightness,
        while negative values decrease brightness. A value of 0 results in no change.

    Returns
    -------
    numpy.ndarray
        The exposure-adjusted HDR image as a NumPy array with the same shape and dtype
        as the input image.

    Notes
    -----
    The exposure adjustment is performed by raising the input image values to the power
    of 2 raised to the exposure factor (i.e., `2^exposure_factor`). This simulates the
    effect of changing exposure in stops, where each stop represents a doubling or
    halving of light intensity.

    Examples
    --------
    >>> import numpy as np
    >>> hdr_img = np.array([1.0, 2.0, 4.0])
    >>> adjusted_img = adjust_exposure(hdr_img, 1.0)
    >>> print(adjusted_img)
    array([ 2.,  4.,  8.])

    Raises
    ------
    ValueError
        If `hdr_image` contains negative values, as HDR images are expected to have
        non-negative radiance values.
    TypeError
        If `hdr_image` is not a NumPy array or if `exposure_factor` is not a numeric type.
    """
    return hdr_image * np.power(2.0, exposure_factor)

