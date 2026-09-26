import os
import sys
import numpy as np
import pandas as pd
print("СТАРТ ПРОГРАМИ: АНАЛІЗ ТЕРИТОРІЇ СЕНСОРІВ", flush=True)
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data_files")
results_dir = os.path.join(base_dir, "results")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

csv_input = os.path.join(data_dir, "sensors.csv")
prob_out = os.path.join(results_dir, "sensors_problems.csv")
anal_out = os.path.join(results_dir, "sensors_analysis.csv")
sect_out = os.path.join(results_dir, "all_sectors.csv")
if not os.path.exists(csv_input) or os.path.getsize(csv_input) == 0:
    print(f"Файлу {csv_input} не знайдено. Генеруємо матрицю 120х140...", flush=True)
    mock_matrix = np.random.uniform(14.0, 26.0, (120, 140))
    mock_matrix[0, 0] = np.nan
    mock_matrix[15, 15] = 99.0
    pd.DataFrame(mock_matrix).to_csv(csv_input, header=False, index=False)
    print("Файл data_files/sensors.csv успішно згенеровано.", flush=True)

try:
    print("\n[Крок 1] Зчитування даних та побудова 4D View...", flush=True)
    df = pd.read_csv(csv_input, header=None)
    data = np.ascontiguousarray(df.to_numpy(dtype=float))
    sectors = data.reshape(12, 10, 14, 10).transpose(0, 2, 1, 3)
    print(f"-> Структура 4D View сформована. Форма: {sectors.shape}", flush=True)
    print("\nАналіз проблемних сенсорів (NaN та відхилення)...", flush=True)
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
                "sector_index": sector_index, "nan_count": nan_count,
                "too_low_count": too_low_count, "too_high_count": too_high_count,
                "invalid_count": invalid_count
            })
            sector_index += 1
    pd.DataFrame(problems_report).to_csv(prob_out, index=False)
    print(f"Файл збережено: {prob_out}", flush=True)
    print("\nФільтрація аномалій більше ніж 10°C від медіани...", flush=True)
    for r in range(12):
        for c in range(14):
            median_val = np.nanmedian(sectors[r, c])
            if not np.isnan(median_val):
                anomalies_mask = (sectors[r, c] < (median_val - 10.0)) | (sectors[r, c] > (median_val + 10.0))
                sectors[r, c, anomalies_mask] = median_val
    print("-> Усі аномальні показники замінено на локальні медіани секторів.", flush=True)
    print("\nРозрахунок температурних метрик секторів...", flush=True)
    analysis_results = []
    sector_number = 0
    for r in range(12):
        for c in range(14):
            sector = sectors[r, c]
            min_temp = np.nanmin(sector)
            max_temp = np.nanmax(sector)
            analysis_results.append({
                "sector_index": sector_number,
                "min_temperature": min_temp,
                "max_temperature": max_temp,
                "mean_temperature": np.nanmean(sector),
                "median_temperature": np.nanmedian(sector),
                "temperature_range": np.nan if np.isnan(max_temp) or np.isnan(min_temp) else (max_temp - min_temp)
            })
            sector_number += 1
    pd.DataFrame(analysis_results).to_csv(anal_out, index=False)
    print(f"Файл збережено: {anal_out}", flush=True)
    print("\nРанжування території (Топ 10 холодних, 10 теплих, 30 середніх)...", flush=True)
    df_valid = pd.DataFrame(analysis_results).dropna().copy()
    global_mean = df_valid["mean_temperature"].mean()

    coldest = df_valid.nsmallest(10, "mean_temperature").copy()
    coldest["category"] = "10 найхолодніших"

    warmest = df_valid.nlargest(10, "mean_temperature").copy()
    warmest["category"] = "10 найтепліших"

    df_remaining = df_valid[
        ~df_valid["sector_index"].isin(set(coldest["sector_index"]).union(set(warmest["sector_index"])))].copy()
    df_remaining["deviation"] = (df_remaining["mean_temperature"] - global_mean).abs()
    closest_to_mean = df_remaining.nsmallest(30, "deviation").copy()
    closest_to_mean["category"] = "30 близькі до середньої температури"

    final_df = pd.concat([coldest, warmest, closest_to_mean]).sort_values(by="sector_index")
    final_df[["sector_index", "category"]].to_csv(sect_out, index=False)
    print(f"Файл збережено: {sect_out}", flush=True)
    print("УСПІХ: ВСІ РОЗРАХУНКИ ЗАВЕРШЕНО ТА ЗБЕРЕЖЕНО!", flush=True)

except Exception as error:
    print(f"\nКРИТИЧНИЙ ЗБІЙ ПІД ЧАС ВИКОНАННЯ: {error}", flush=True)