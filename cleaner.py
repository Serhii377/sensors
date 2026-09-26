import numpy as np


def clean_sector(all_sectors: np.ndarray, r: int, c: int) -> None:
    median_val = np.nanmedian(all_sectors[r, c])
    if np.isnan(median_val):
        return

    too_low = all_sectors[r, c] < (median_val - 10.0)
    too_high = all_sectors[r, c] > (median_val + 10.0)
    anomalies_mask = too_low | too_high
    all_sectors[r, c, anomalies_mask] = median_val


def clean_all_sectors(all_sectors: np.ndarray) -> None:
    for r in range(12):
        for c in range(14):
            clean_sector(all_sectors, r, c)