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


def melt_dataframe(df):
    """
    Melts the given DataFrame into the desired format for submission.
    Each column (C1 to C7) is transformed into rows with a corresponding prediction type and fracture target.
    """
    # Melt the DataFrame: transform columns (C1 to C7) into rows
    melted_df = df.melt(
        id_vars=["StudyInstanceUID", "patient_overall"],  # Keep these columns as is
        value_vars=["C1", "C2", "C3", "C4", "C5", "C6", "C7"],  # Columns to unpivot
        var_name="prediction_type",  # Name of the new column for C1-C7
        value_name="fracture_target"  # Name of the new column for target values
    )
    # Create the row_id column by combining StudyInstanceUID and prediction_type
    melted_df["row_id"] = melted_df["StudyInstanceUID"] + "_" + melted_df["prediction_type"]
    # Add an empty column for predictions
    melted_df["fracture_prediction"] = ""
    # Rearrange columns for the desired output format
    melted_df = melted_df[
        ["row_id", "StudyInstanceUID", "prediction_type", "fracture_prediction", "fracture_target"]
    ]
    return melted_df


def split_train_test(train_csv_path, train_images_path, test_images_path, test_csv_path):
    """
    Splits the `train_images` folder into 80% training and 20% testing.
    Moves the testing folders to `test_images` and creates a new `test.csv` in the desired melted format.

    Parameters:
    - train_csv_path: Path to the training CSV file.
    - train_images_path: Path to the directory containing training images (folders per StudyInstanceUID).
    - test_images_path: Path to the directory where test images will be moved.
    - test_csv_path: Path to save the newly created test CSV file.
    """
    # Get the list of StudyInstanceUID folders in the training images directory
    study_folders = os.listdir(train_images_path)
    if not study_folders:
        raise ValueError("No folders found in train_images.")

    # Total number of folders in train_images
    train_images_num = len(study_folders)
    print(f"Folders in train_images: {train_images_num}, {int(train_images_num * 0.2)} folders will be moved to test_images.")

    # Shuffle the list of folders to randomize train-test split
    random.shuffle(study_folders)
    # Calculate the split index for 20% testing
    split_index = int(0.2 * train_images_num)
    # Divide folders into test and train sets
    test_study_folders = study_folders[:split_index]  # 20% for test
    train_study_folders = study_folders[split_index:]  # Remaining 80% for training

    # Create the test_images directory if it doesn't exist
    os.makedirs(test_images_path, exist_ok=True)
    # Move test folders to the test_images directory
    for folder in test_study_folders:
        shutil.move(os.path.join(train_images_path, folder), os.path.join(test_images_path, folder))
    print(f"Moved {len(test_study_folders)} folders to {test_images_path}.")

    # Read the train.csv file into a DataFrame
    train_df = pd.read_csv(train_csv_path)

    # Filter the DataFrame to keep only training records
    train_df_filtered = train_df[train_df["StudyInstanceUID"].isin(train_study_folders)]
    # Save the updated training DataFrame back to train.csv
    train_df_filtered.to_csv(train_csv_path, index=False)
    print(f"Updated {train_csv_path} with {len(train_df_filtered)} training records.")

    # Filter the DataFrame to create a test DataFrame
    test_df = train_df[train_df["StudyInstanceUID"].isin(test_study_folders)]
    # Transform the test DataFrame to the desired melted format
    melted_test_df = melt_dataframe(test_df)
    # Save the melted test DataFrame to test.csv
    melted_test_df.to_csv(test_csv_path, index=False)
    print(f"Created {test_csv_path} with {len(melted_test_df)} records in the desired format.")

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
