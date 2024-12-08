import os
import pandas as pd
from utils.data_processor import split_train_test, load_dicom
import shutil
import subprocess

def setup_kaggle():
    """
    Set up Kaggle API credentials.
    """
    if not os.path.exists("kaggle.json"):
        raise FileNotFoundError("kaggle.json not found! Place it in the same directory as this script.")
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    shutil.copy("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)


def download_and_extract():
    """
    Download the dataset from Kaggle and extract it into the `dataset` directory.
    Remove the zip file afterward.
    """

    print("Downloading dataset...")
    subprocess.run(["kaggle", "competitions", "download", "-c", "rsna-2022-cervical-spine-fracture-detection"], check=True)
    print("Extracting dataset...")
    os.makedirs("dataset", exist_ok=True)
    subprocess.run(["unzip", "-q", "rsna-2022-cervical-spine-fracture-detection.zip", "-d", "dataset"], check=True)
    os.remove("rsna-2022-cervical-spine-fracture-detection.zip")
    print("Dataset downloaded and extracted into 'dataset' folder.")


def clean_up_folders():
    """
    Remove unnecessary files and folders from the dataset directory.
    """
    paths_to_remove = [
        "dataset/test_images",
        "dataset/test.csv",
        "dataset/sample_submission.csv",
        "dataset/train_bounding_boxes.csv",
        "dataset/segmentations"
    ]
    for path in paths_to_remove:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    print("Unnecessary files and folders removed.")


if __name__ == "__main__":
    try:
        # Step 1: Setup Kaggle API
        setup_kaggle()

        # Step 2: Download and extract dataset
        download_and_extract()

        # Step 3: Remove unnecessary files and folders
        clean_up_folders()

        # Step 4: Split train and test using the imported function
        split_train_test(
            train_csv_path="dataset/train.csv",
            train_images_path="dataset/train_images",
            test_images_path="dataset/test_images",
            test_csv_path="dataset/test.csv"
        )

        print("Dataset processing complete.")

    except Exception as e:
        print("Error:", str(e))
