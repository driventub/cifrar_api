from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from Crypto.Cipher import AES
import cv2
import os
from Crypto.Util.Padding import pad, unpad
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from starlette.responses import StreamingResponse

app = FastAPI()

def encrypt_image(image_path: str, key: bytes) -> bytes:
    """
    Encrypts an image using AES and returns the ciphertext as a byte array.

    Args:
        image_path: Path to the image to be encrypted.
        key: 16-byte encryption key.

    Returns:
        ciphertext: Encrypted image data as a byte array.
    """
    # Read image
    img = cv2.imread(image_path)
    # Flatten image to a 1D array
    img_flat = img.flatten()
    # Pad the flattened image data
    padded_img = pad(img_flat.tobytes(), AES.block_size)
    # Create an AES cipher object
    cipher = AES.new(key, AES.MODE_ECB)
    # Encrypt the padded image data
    ciphertext = cipher.encrypt(padded_img)
    return ciphertext



@app.post("/encrypt-image/")
async def encrypt_image_api(image_file: UploadFile = File(...), key = os.urandom(16)):
    """
    Endpoint to encrypt an image using AES encryption.

    Args:
        image_file: The image file to be encrypted.
        key: The encryption key (16 bytes).

    Returns:
        Dict: A dictionary containing the encrypted image data.
    """
    try:
        if not key or len(key) != 16:
            raise HTTPException(status_code=400, detail="Invalid encryption key. Key must be 16 bytes long.")

        # Create a temporary file to save the uploaded image
        with open("temp_image.jpg", "wb") as temp_image:
            temp_image.write(await image_file.read())

        # Encrypt the uploaded image
        ciphertext = encrypt_image("temp_image.jpg", key)

        # Remove the temporary image file
        os.remove("temp_image.jpg")

        return {"encrypted_image": ciphertext.hex()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def apply_filter(image: Image.Image):
    # Convert the PIL image to a NumPy array
    img_array = np.array(image)
    # Apply the filter (set green and blue channels to 0)
    img_array[:, :, 1] = 0  # Green channel
    img_array[:, :, 2] = 0  # Blue channel
    # Convert the filtered NumPy array back to a PIL image
    filtered_image = Image.fromarray(img_array)
    return filtered_image

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # Read the contents of the uploaded file
    contents = await file.read()
    # Open the image using PIL
    img = Image.open(BytesIO(contents))
    # Apply the filter
    filtered_img = apply_filter(img)
    # Save the filtered image to a BytesIO buffer
    img_bytes = BytesIO()
    filtered_img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    # Return the filtered image bytes as a response
    return StreamingResponse(img_bytes, media_type="image/jpeg")


