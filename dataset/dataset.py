import os
import shutil
import subprocess
import random
import pandas as pd


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


def split_train_test(train_csv_path, train_images_path, test_images_path, test_csv_path):
    """
    Splits the `train_images` into 80% training and 20% testing.
    Moves the 20% test folders to the `test_images` directory and creates a new `test.csv`.

    Parameters:
    - train_csv_path: Path to the `train.csv` file.
    - train_images_path: Path to the directory containing training images (organized by StudyInstanceUID folders).
    - test_images_path: Path to the directory where test images will be moved.
    - test_csv_path: Path to save the newly created `test.csv`.
    """
    # List all StudyInstanceUID folders in train_images
    study_folders = os.listdir(train_images_path)
    if not study_folders:
        raise ValueError("No folders found in train_images.")

    train_images_num = len(study_folders)  # Total number of folders in train_images
    print(f"Folders in train_images: {train_images_num}, {int(train_images_num * 0.2)} folders will be moved to test_images.")

    # Shuffle and split folders into 80% train and 20% test
    random.shuffle(study_folders)  # Randomly shuffle the list of folders
    split_index = int(0.2 * train_images_num)
    test_study_folders = study_folders[:split_index]  # 20% for test
    train_study_folders = study_folders[split_index:]  # Remaining 80% for training

    # Move test_study_folders to test_images directory
    os.makedirs(test_images_path, exist_ok=True)
    for folder in test_study_folders:
        shutil.move(os.path.join(train_images_path, folder), os.path.join(test_images_path, folder))
    print(f"Moved {len(test_study_folders)} folders to {test_images_path}.")

    # Read the train.csv file
    train_df = pd.read_csv(train_csv_path)

    # Filter train.csv to keep only training StudyInstanceUIDs
    train_df_filtered = train_df[train_df["StudyInstanceUID"].isin(train_study_folders)]
    train_df_filtered.to_csv(train_csv_path, index=False)  # Overwrite train.csv with filtered data
    print(f"Updated {train_csv_path} with {len(train_df_filtered)} training records.")

    # Create test.csv with test StudyInstanceUIDs
    test_df = train_df[train_df["StudyInstanceUID"].isin(test_study_folders)]
    test_df.to_csv(test_csv_path, index=False)
    print(f"Created {test_csv_path} with {len(test_df)} test records.")

    print(f"Train-Test split completed: {len(train_study_folders)} train, {len(test_study_folders)} test.")


if __name__ == "__main__":
    try:
        # Step 1: Setup Kaggle API
        setup_kaggle()

        # Step 2: Download and extract dataset
        download_and_extract()

        # Step 3: Remove unnecessary files and folders
        clean_up_folders()

        # Step 4: Split train and test
        split_train_test(
            train_csv_path="dataset/train.csv",
            train_images_path="dataset/train_images",
            test_images_path="dataset/test_images",
            test_csv_path="dataset/test.csv"
        )

        print("Dataset processing complete.")

    except Exception as e:
        print("Error:", str(e))
