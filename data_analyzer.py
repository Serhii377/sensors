import os
import numpy as np
import pandas as pd
from cleaner import clean_all_sectors
from data_service import load_sensor_data, create_sectors_4d_view


def analyze_sector(all_sectors: np.ndarray, r: int, c: int, sector_number: int) -> dict:
    sector = all_sectors[r, c]
    min_temp = np.nanmin(sector)
    max_temp = np.nanmax(sector)
    mean_temp = np.nanmean(sector)
    median_temp = np.nanmedian(sector)
    temp_range = np.nan if np.isnan(max_temp) or np.isnan(min_temp) else (max_temp - min_temp)

    return {
        "sector_index": sector_number,
        "min_temperature": min_temp,
        "max_temperature": max_temp,
        "mean_temperature": mean_temp,
        "median_temperature": median_temp,
        "temperature_range": temp_range
    }


def process_data_sectors(all_sectors: np.ndarray, output_path: str = "results/sensors_analysis.csv") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    analysis_results = []
    sector_number = 0

    for r in range(12):
        for c in range(14):
            analysis_results.append(analyze_sector(all_sectors, r, c, sector_number))
            sector_number += 1

    pd.DataFrame(analysis_results).to_csv(output_path, index=False)


if __name__ == "__main__":
    try:
        data = load_sensor_data("data_files/sensors.csv")
        all_sectors = create_sectors_4d_view(data)
        clean_all_sectors(all_sectors)
        process_data_sectors(all_sectors)
        print("Файл sensors_analysis.csv успішно створено.")
    except Exception as e:
        print(f"Помилка запуску: {e}")