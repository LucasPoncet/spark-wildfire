"""Painting an RGB array into the page without going through a file.

`st.image` hands a numpy array to Streamlit's media manager, which writes a
file and serves it by URL. Repainting many frames a second creates and
collects those files faster than the browser fetches them, and the console
fills with 404s on media that no longer exists. Encoding each frame as a data
URI keeps the whole image inside the message, so there is nothing to collect.

`st.html` rather than `st.markdown`: the markdown renderer drops the style
attribute, which leaves a 121-pixel grid drawn at 121 pixels instead of filling
the column.
"""

import base64
import io

import matplotlib.image
import numpy as np
import numpy.typing as npt
from streamlit.delta_generator import DeltaGenerator

IMAGE_FORMAT: str = "png"


def encode_rgb_image_as_data_uri(image: npt.NDArray[np.uint8]) -> str:
    """Encode an RGB image as a base64 PNG data URI.

    Args:
        image: Uint8 array of shape (height, width, 3).

    Returns:
        A `data:image/png;base64,...` URI.
    """
    buffer = io.BytesIO()
    matplotlib.image.imsave(buffer, image, format=IMAGE_FORMAT)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{IMAGE_FORMAT};base64,{encoded}"


def paint_rgb_image(placeholder: DeltaGenerator, image: npt.NDArray[np.uint8]) -> None:
    """Draw one RGB frame into a placeholder, full width.

    Args:
        placeholder: The `st.empty()` slot to draw into.
        image: Uint8 array of shape (height, width, 3).
    """
    placeholder.html(
        f'<img src="{encode_rgb_image_as_data_uri(image)}" '
        'style="display:block;width:100%;height:auto;'
        'image-rendering:pixelated;border-radius:4px">'
    )
