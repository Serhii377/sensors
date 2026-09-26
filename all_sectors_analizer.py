import os
import numpy as np
import pandas as pd


def categorize_sectors(all_sectors: np.ndarray, output_path: str = "results/all_sectors.csv") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sector_data = []
    sector_index = 0

    for r in range(12):
        for c in range(14):
            sector_data.append({
                "sector_index": sector_index,
                "mean_temperature": np.nanmean(all_sectors[r, c])
            })
            sector_index += 1

    df_valid = pd.DataFrame(sector_data).dropna().copy()
    global_mean = df_valid["mean_temperature"].mean()

    coldest = df_valid.nsmallest(10, "mean_temperature").copy()
    coldest["category"] = "10 найхолодніших"

    warmest = df_valid.nlargest(10, "mean_temperature").copy()
    warmest["category"] = "10 найтепліших"

    used_indices = set(coldest["sector_index"]).union(set(warmest["sector_index"]))
    df_remaining = df_valid[~df_valid["sector_index"].isin(used_indices)].copy()
    df_remaining["deviation"] = (df_remaining["mean_temperature"] - global_mean).abs()

    closest_to_mean = df_remaining.nsmallest(30, "deviation").copy()
    closest_to_mean["category"] = "30 близькі до середньої температури"

    final_df = pd.concat([coldest, warmest, closest_to_mean]).sort_values(by="sector_index")
    final_df[["sector_index", "category"]].to_csv(output_path, index=False)