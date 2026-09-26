import os
import numpy as np
import pandas as pd


def analyze_sensor_problems(sectors: np.ndarray, output_path: str = "results/sensors_problems.csv") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    problems_report = []
    sector_index = 0

    for r in range(12):
        for c in range(14):
            sector = sectors[r, c]
            nan_count = int(np.isnan(sector).sum())
            median_val = np.nanmedian(sector)

            if np.isnan(median_val):
                too_low_count, too_high_count = 0, 0
                invalid_count = nan_count
            else:
                too_low_count = int(np.nansum(sector < (median_val - 10.0)))
                too_high_count = int(np.nansum(sector > (median_val + 10.0)))
                invalid_count = nan_count + too_low_count + too_high_count

            problems_report.append({
                "sector_index": sector_index,
                "nan_count": nan_count,
                "too_low_count": too_low_count,
                "too_high_count": too_high_count,
                "invalid_count": invalid_count
            })
            sector_index += 1

    pd.DataFrame(problems_report).to_csv(output_path, index=False)