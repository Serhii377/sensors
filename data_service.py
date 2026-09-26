import os
import numpy as np
import pandas as pd

def load_sensor_data(file_path: str = "data_files/sensors.csv") -> np.ndarray:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не знайдено за шляхом: {file_path}")
    df = pd.read_csv(file_path, header=None)
    return np.ascontiguousarray(df.to_numpy(dtype=float))

def create_sectors_4d_view(data: np.ndarray) -> np.ndarray:
    return data.reshape(12, 10, 14, 10).transpose(0, 2, 1, 3)