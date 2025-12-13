import os
import zipfile
import gdown

MODEL_DIR = "models"
ZIP_PATH = "models.zip"
DRIVE_FILE_ID = "1q0Ez5qk7YPJAzFd0SJ_xYmC1TqGMEjoy"
def ensure_models():
    if os.path.exists(MODEL_DIR):
        return

    print("Downloading models from Google Drive...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, ZIP_PATH, quiet=False)

    print("Extracting models...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(".")

    os.remove(ZIP_PATH)
    print("Models ready.")
