from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def encrypt_image(image_path, key):
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
    # plt.imshow(img)
    # plt.title('Imagen Original')
    # plt.show()


    # Flatten image to a 1D array
    img_flat = img.flatten()

    # Pad the flattened image data
    padded_img = pad(img_flat.tobytes(), AES.block_size)

    # Create an AES cipher object
    cipher = AES.new(key, AES.MODE_ECB)

    # Encrypt the padded image data
    ciphertext = cipher.encrypt(padded_img)

    return ciphertext

def decrypt_image(ciphertext, key, original_shape):
    """
    Decrypts the image data using AES and returns the original image.

    Args:
        ciphertext: Encrypted image data as a byte array.
        key: 16-byte encryption key.
        original_shape: Shape of the original image.

    Returns:
        decrypted_image: Decrypted image data as a NumPy array.
    """
    # Create an AES cipher object
    cipher = AES.new(key, AES.MODE_ECB)

    # Decrypt the ciphertext
    decrypted_data = cipher.decrypt(ciphertext)

    # Unpad the decrypted data
    unpadded_data = unpad(decrypted_data, AES.block_size)

    # Convert the unpadded data back to a NumPy array
    decrypted_image = np.frombuffer(unpadded_data, dtype=np.uint8)

    # Reshape the decrypted image to its original shape
    decrypted_image = decrypted_image.reshape(original_shape)

    return decrypted_image

key = os.urandom(16)# Replace with your actual key
ciphertext = encrypt_image("imagen_resultante.jpg", key)
decrypted_image = decrypt_image(ciphertext, key, cv2.imread("imagen_resultante.jpg").shape)
cv2.imwrite("perro_encrip.jpg", decrypted_image)