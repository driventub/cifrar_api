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

def apply_filter(img1: Image.Image, img2: Image.Image):
    # Convert the PIL image to a NumPy array
    img_array = np.array(img1)
    # Apply the filter (set green and blue channels to 0)
    img_array[:, :, 1] = 0  # Green channel
    img_array[:, :, 2] = 0  # Blue channel
    # Convert the filtered NumPy array back to a PIL image
    imagen_filtrada1 = Image.fromarray(img_array)
    
    img_array2 = np.array(img2)
    # Apply the filter (set green and blue channels to 0)
    img_array2[:, :, 0] = 0  # Green channel
    img_array2[:, :, 1] = 0  # Blue channel
    # Convert the filtered NumPy array back to a PIL image
    imagen_filtrada2 = Image.fromarray(img_array2)

    matriz_imagen = np.array(imagen_filtrada1)

    max_columnas = matriz_imagen.shape[1]
    secuencia_aleatoria = np.random.permutation(max_columnas) + 1

    # Imprime la secuencia generada
    print("Secuencia de valores aleatorios sin repeticiones:", secuencia_aleatoria)

    # Usa la secuencia de valores aleatorios para transponer las columnas
    matriz_transpuesta = matriz_imagen[:, secuencia_aleatoria - 1]

    #mostrar la imagen transpuesta
    imagen_transpuesta = Image.fromarray(matriz_transpuesta)
    imagen_transpuesta.save("imagen_transpuesta.jpg")

    # Guarda la matriz original y secuencia original
    matriz_original = matriz_transpuesta.copy()
    secuencia_aleatoria_original = secuencia_aleatoria.copy()

    # Transpone nuevamente las columnas con la secuencia aleatoria invertida
    secuencia_aleatoria_invertida = np.argsort(secuencia_aleatoria_original)
    matriz_original = matriz_original[:, secuencia_aleatoria_invertida]
    imagen_original = Image.fromarray(matriz_original)

    matriz_segunda_imagen = np.array(imagen_filtrada2)

# Obtener la forma de la matriz transpuesta
    filas_transpuesta, columnas_transpuesta, _ = matriz_transpuesta.shape

    # Obtener las dimensiones de la segunda imagen
    filas_segunda_imagen, columnas_segunda_imagen, _ = matriz_segunda_imagen.shape

    # Ajustar las dimensiones de la segunda imagen si es necesario
    if filas_transpuesta > filas_segunda_imagen or columnas_transpuesta > columnas_segunda_imagen:
        nueva_filas = max(filas_transpuesta, filas_segunda_imagen)
        nueva_columnas = max(columnas_transpuesta, columnas_segunda_imagen)
        segunda_imagen = imagen_filtrada2.resize((nueva_columnas, nueva_filas))

        # Actualizar la matriz de la segunda imagen
        matriz_segunda_imagen = np.array(segunda_imagen)

    # Iterar sobre las posiciones pares de la segunda imagen y reemplazar con la matriz transpuesta
    for i in range(filas_transpuesta):
        for j in range(columnas_transpuesta):
            if i % 2 == 0 and j % 2 == 0:  # Si la fila y la columna son pares
                matriz_segunda_imagen[i, j] = matriz_transpuesta[i, j]

    # Crear una nueva imagen a partir de la matriz modificada
    nueva_imagen = Image.fromarray(matriz_segunda_imagen)
    return nueva_imagen

@app.post("/uploadfile/")
async def create_upload_file(img1: UploadFile,img2: UploadFile ):
    # Read the contents of the uploaded img1
    cont1 = await img1.read()
    cont2 = await img2.read()
    # Open the image using PIL
    image1 = Image.open(BytesIO(cont1))
    image2 = Image.open(BytesIO(cont2))
    # Apply the filter
    filtered_img = apply_filter(image1,image2)
    # Save the filtered image to a BytesIO buffer
    img_bytes = BytesIO()
    filtered_img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    # Return the filtered image bytes as a response
    return StreamingResponse(img_bytes, media_type="image/jpeg")


