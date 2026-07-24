import os
import zipfile
import gdown
import shutil

MODEL_DIR = "models"
ZIP_PATH = "models.zip"

def ensure_models():
    if os.path.exists(MODEL_DIR) and os.path.isdir(MODEL_DIR):
        if len(os.listdir(MODEL_DIR)) > 0:
            print(f"✅ '{MODEL_DIR}' directory found. Skipping download.")
            return

    print("--- STARTING MODEL DOWNLOAD ---")
    
    url = "https://drive.google.com/file/d/1q0Ez5qk7YPJAzFd0SJ_xYmC1TqGMEjoy"
    output = gdown.download(url, ZIP_PATH, quiet=False, fuzzy=True)

    if not os.path.exists(ZIP_PATH):
        print("❌ Error: ZIP file download failed.")
        return

    print("Extracting models...")
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            file_list = zip_ref.namelist()
            if not file_list:
                print("❌ Error: Zip file is empty.")
                return
            
            first_item = file_list[0]
            
            if first_item.startswith("models/") or first_item.startswith("models\\"):
                print("Detected 'models' folder in zip. Extracting to root...")
                zip_ref.extractall(".")
            else:
                print("Detected flat zip. Creating 'models' folder and extracting inside...")
                os.makedirs(MODEL_DIR, exist_ok=True)
                zip_ref.extractall(MODEL_DIR)
                
        print("✅ Models ready.")

    except zipfile.BadZipFile:
        print("❌ Error: The downloaded file is not a valid zip file.")
    except Exception as e:
        print(f"❌ Extraction error: {e}")
    finally:
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
