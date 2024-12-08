import os
import glob
import re
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config  
from utils.utils import load_model, load_dicom  
from utils.data_processor import DataSet 
from models.fracturenet import fracturenet 

def predict(models, dataset, max_batches=1e9):
    """Generate predictions using an ensemble of models."""
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())
    predictions = []
    with torch.no_grad():
        for idx, X in enumerate(tqdm(dataloader, desc="Predicting")):
            pred = torch.zeros(len(X), 14).to(Config.DEVICE)
            for m in models:
                y1, y2 = m.predict(X.to(Config.DEVICE))
                pred += torch.concat([y1, y2], dim=1) / len(models)
            predictions.append(pred)
    return torch.concat(predictions).cpu().numpy()


def patient_prediction(df):
    """Aggregate predictions at the patient level."""
    c1c7 = np.average(df[Config.FRAC_COLS].values, axis=0, weights=df[Config.VERT_COLS].values)
    pred_patient_overall = 1 - np.prod(1 - c1c7)
    return pd.Series(data=np.concatenate([[pred_patient_overall], c1c7]), index=['patient_overall'] + [f'C{i}' for i in range(1, 8)])


if __name__ == "__main__":
    # Ensure required files and directories exist
    if not os.path.exists("./dataset/test.csv"):
        raise FileNotFoundError("Test file './dataset/test.csv' not found.")
    for name in Config.MODEL_NAMES:
        checkpoint_path = os.path.join(Config.CHECKPOINTS_PATH, f"{name}.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")

    print("Loading test data...")
    df_test = pd.read_csv("./dataset/test.csv")
    print("Test data loaded.")

    print("Preparing test dataset...")
    test_slices = glob.glob(f"{Config.TEST_IMAGES_PATH}/*/*")
    test_slices = [re.findall(f"{Config.TEST_IMAGES_PATH}/(.*)/(.*).dcm", s)[0] for s in test_slices]
    df_test_slices = pd.DataFrame(data=test_slices, columns=['StudyInstanceUID', 'Slice']).astype({'Slice': int})
    df_test_slices = df_test_slices.sort_values(['StudyInstanceUID', 'Slice']).reset_index(drop=True)
    ds_test = DataSet(df_test_slices, Config.TEST_IMAGES_PATH, transforms=None)
    print("Test dataset prepared.")

    print("Loading models...")
    models = [load_model(fracturenet(), name, Config.CHECKPOINTS_PATH).to(Config.DEVICE) for name in Config.MODEL_NAMES]
    print("Models loaded.")

    print("Generating predictions...")
    pred = predict(models, ds_test)
    print("Predictions completed.")

    print("Preparing prediction DataFrame...")
    df_pred = pd.DataFrame(data=pred, columns=Config.FRAC_COLS + Config.VERT_COLS)
    df_test_pred = pd.concat([df_test_slices, df_pred], axis=1).sort_values(['StudyInstanceUID', 'Slice'])
    df_patient_pred = df_test_pred.groupby('StudyInstanceUID').apply(lambda df: patient_prediction(df))

    print("Merging predictions with test set...")
    df_test = df_test.set_index('StudyInstanceUID').join(df_patient_pred)
    df_test['fracture_prediction'] = df_test.apply(lambda r: r[r.prediction_type], axis=1)

    print("Generating classification report...")
    final = df_test[['row_id', 'fracture_prediction', 'fracture_target']]
    final.loc[:, 'binary_prediction'] = (final['fracture_prediction'] >= 0.5).astype(int)
    report = classification_report(final['fracture_target'], final['binary_prediction'], target_names=['No Fracture', 'Fracture'], output_dict=True)
    report['AUC-ROC'] = {'score': roc_auc_score(final['fracture_target'], final['fracture_prediction'])}

    # Display report
    report_df = pd.DataFrame(report).T
    print(report_df)