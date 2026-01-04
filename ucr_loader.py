import os
import numpy as np
import requests
import zipfile
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

UCR_URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"

def download_ucr_dataset(name="Plane", root="./data"):
    """
    Downloads the full UCRArchive_2018 if missing.
    Extracts only the dataset represented by name.
    """
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "UCRArchive_2018.zip")
    extract_path = os.path.join(root, "UCRArchive_2018")

    if not os.path.exists(zip_path):
        print("Downloading UCR Archive (~600MB)...")
        r = requests.get(UCR_URL, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    if not os.path.exists(extract_path):
        print("Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(root)

    return os.path.join(extract_path, name)


def load_dataset(name="Plane", root=".\data"):
    path = download_ucr_dataset(name, root)
 
    train_file = os.path.join(path, name+"_TRAIN.tsv")
    test_file = os.path.join(path, name+"_TEST.tsv")

    train = np.loadtxt(train_file)
    test  = np.loadtxt(test_file)

    # First column = label
    X_train = train[:, 1:]
    y_train = train[:, 0].astype(int)

    X_test = test[:, 1:]
    y_test = test[:, 0].astype(int)

    # Normalize
    scaler_train = StandardScaler()
    X_train = scaler_train.fit_transform(X_train.T).T
    scaler_test = StandardScaler()
    X_test = scaler_test.fit_transform(X_test.T).T

    return X_train, y_train, X_test, y_test


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return idx, self.X[idx].unsqueeze(-1), self.y[idx] 
