from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
from io import BytesIO

app = FastAPI()

def apply_red_filter(image_bytes: bytes) -> bytes:
    """
    Applies a red filter to the image bytes and returns the filtered image bytes.

    Args:
        image_bytes: Bytes of the input image.

    Returns:
        bytes: Bytes of the filtered image.
    """
    try:
        # Open the image from bytes
        image = Image.open(BytesIO(image_bytes))
        # Convert image to NumPy array
        image_array = np.asarray(image)
        # Create a copy of the image array
        filtered_image_array = np.copy(image_array)
        # Set green and blue channels to 0 (zero)
        filtered_image_array[:, :, 1] = 0
        filtered_image_array[:, :, 2] = 0
        # Create a new image from the filtered image array
        filtered_image = Image.fromarray(filtered_image_array)
        # Convert the filtered image to bytes
        with BytesIO() as output:
            filtered_image.save(output, format='JPEG')
            filtered_image_bytes = output.getvalue()
        return filtered_image_bytes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/apply-red-filter/")
async def apply_red_filter_api(image_file: UploadFile = File(...)):
    """
    Endpoint to apply a red filter to an image.

    Args:
        image_file: The input image file.

    Returns:
        bytes: Bytes of the filtered image.
    """
    try:
        # Read the image bytes as binary data
        image_bytes = await image_file.read()
        # Apply the red filter to the uploaded image
        filtered_image_bytes = apply_red_filter(image_bytes)
        return filtered_image_bytes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
