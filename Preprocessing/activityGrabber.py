import os
import pandas as pd
import numpy as np

patient = "001"
directory = f"/Volumes/Untitled/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2/{patient}/combined"

for filename in os.listdir(directory):
    data = pd.read_csv(os.path.join(directory, filename), encoding='latin-1')
    data["datetime"] = pd.to_datetime(data["datetime"])
    
    data20 = data.resample("20min", on="datetime").mean().reset_index()
    
    # Identify periods where values are above the 3Q
    above3Q = pd.DataFrame()
    above3Q["datetime"] = data20["datetime"]
    above3Q["heart rate"] = data20[" hr"] > data[" hr"].quantile(0.75)
    above3Q["absDistance"] = data20["absDistance"] > data20["absDistance"].quantile(0.75)

    above3Q["all"] = above3Q[[f"{col}" for col in above3Q.columns if col != "datetime"]].all(axis=1)
    above3Q["all_start"] = (above3Q["all"] & ~above3Q["all"].shift(1, fill_value=False))
    above3Q["all_end"] = (~above3Q["all"] & above3Q["all"].shift(1, fill_value=False))

    # array to store periods of activity
    above = []

    # Find the start and end points of periods
    starts = above3Q.loc[above3Q["all_start"], "datetime"]
    ends = above3Q.loc[above3Q["all_end"], "datetime"]

    # Last row ends period if it doesn"t end within the dataset
    if len(starts) > len(ends):
            ends = ends.append(pd.Series(above3Q["datetime"].iloc[-1]))

    for start_time, end_time in zip(starts, ends):
            if start_time <= pd.Timestamp(start_time.date()) + pd.Timedelta(hours=6):
                pass
            else:
                above.append((start_time, end_time + pd.Timedelta(minutes=30)))
    
    above = np.array(above)
    start_times = above[:, 0]
    end_times = above[:, 1]

    start_times = pd.to_datetime(start_times)
    end_times = pd.to_datetime(end_times)
    
    if not os.path.exists(os.path.join(directory, "activity")):
        os.makedirs(os.path.join(directory, "activity"))
        
    day = filename.split('_')[1].replace('.csv', '')
    for i, (start, end) in enumerate(zip(start_times, end_times)):
        startTime = start.strftime('%H:%M').replace(":", "-")
        endTime = end.strftime('%H:%M').replace(":", "-")
    # Filter DataFrame based on current period
        activity = data[data['datetime'].between(start, end)]
    
    # Save filtered DataFrame as CSV with index=False to exclude row numbers
        name = f"Day{day}_({startTime},{endTime}).csv"
        file = os.path.join(os.path.join(directory, "activity"), name)
        activity.to_csv(file, index=False)
        print(f"Saved period {i+1} data to '{file}'")
    
