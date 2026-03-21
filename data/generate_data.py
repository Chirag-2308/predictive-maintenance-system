import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
NUM_MACHINES   = 50
DAYS           = 180          # 6 months of data
SAMPLES_PER_DAY = 24          # one reading per hour
FAULT_RATE     = 0.30         # 30% of machines develop faults
RANDOM_SEED    = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

MACHINE_TYPES = [
    "CNC Lathe", "Hydraulic Press", "Conveyor Belt",
    "Compressor", "Pump Station", "Welding Robot",
    "Packaging Unit", "Drilling Machine"
]

LOCATIONS = [
    "Floor A – Bay 1", "Floor A – Bay 2", "Floor B – Bay 1",
    "Floor B – Bay 2", "Floor C – Bay 1", "Maintenance Hall"
]

# ─────────────────────────────────────────────
#  NORMAL OPERATING RANGES  (per machine type)
# ─────────────────────────────────────────────
RANGES = {
    "CNC Lathe":       dict(temp=(55,75),  vibration=(0.5,2.0), rpm=(800,1200),  pressure=(3.0,5.0)),
    "Hydraulic Press": dict(temp=(60,80),  vibration=(1.0,3.0), rpm=(200,400),   pressure=(8.0,12.0)),
    "Conveyor Belt":   dict(temp=(40,60),  vibration=(0.3,1.5), rpm=(100,200),   pressure=(1.5,3.0)),
    "Compressor":      dict(temp=(70,90),  vibration=(1.5,3.5), rpm=(1500,2200), pressure=(6.0,10.0)),
    "Pump Station":    dict(temp=(45,65),  vibration=(0.8,2.5), rpm=(600,1000),  pressure=(4.0,7.0)),
    "Welding Robot":   dict(temp=(80,100), vibration=(2.0,4.0), rpm=(300,600),   pressure=(2.5,4.5)),
    "Packaging Unit":  dict(temp=(35,55),  vibration=(0.2,1.2), rpm=(400,800),   pressure=(1.0,2.5)),
    "Drilling Machine":dict(temp=(60,85),  vibration=(1.8,4.5), rpm=(700,1100),  pressure=(3.5,6.0)),
}

def generate_machine_metadata():
    machines = []
    for i in range(NUM_MACHINES):
        mtype    = random.choice(MACHINE_TYPES)
        machines.append({
            "machine_id":   f"MCH-{i+1:03d}",
            "machine_type": mtype,
            "location":     random.choice(LOCATIONS),
            "age_years":    round(random.uniform(0.5, 12.0), 1),
            "model":        f"Model-{random.choice(['A','B','C','X'])}{random.randint(100,999)}",
        })
    return pd.DataFrame(machines)


def add_fault_signature(series, fault_start_idx, sensor, intensity=1.0):
    """Gradually inject anomalous values starting 48 h before failure."""
    n     = len(series)
    result = series.copy()
    ramp   = np.linspace(0, 1, n - fault_start_idx)

    if sensor == "temperature":
        result[fault_start_idx:] += ramp * 30 * intensity
    elif sensor == "vibration":
        result[fault_start_idx:] += ramp * 6.0 * intensity
    elif sensor == "rpm":
        result[fault_start_idx:] -= ramp * 300 * intensity
        result = np.clip(result, 0, None)
    elif sensor == "pressure":
        result[fault_start_idx:] += ramp * 4.0 * intensity
    return result


def generate_sensor_data(metadata_df):
    total_readings = NUM_MACHINES * DAYS * SAMPLES_PER_DAY
    print(f"Generating {total_readings:,} sensor readings …")

    base_time  = datetime(2024, 1, 1, 0, 0, 0)
    all_records = []

    fault_machines = set(
        random.sample(range(NUM_MACHINES), k=int(NUM_MACHINES * FAULT_RATE))
    )

    for idx, row in metadata_df.iterrows():
        mtype    = row["machine_type"]
        r        = RANGES[mtype]
        n_pts    = DAYS * SAMPLES_PER_DAY
        age_factor = 1 + row["age_years"] * 0.01

        # base signals with slight drift and daily cycles
        t_arr = np.arange(n_pts)
        daily = np.sin(2 * np.pi * t_arr / 24) * 2   # daily thermal cycle

        temp      = np.random.normal(np.mean(r["temp"]),      (r["temp"][1]-r["temp"][0])/6,      n_pts) + daily * 1.5
        vibration = np.random.normal(np.mean(r["vibration"]), (r["vibration"][1]-r["vibration"][0])/6, n_pts)
        rpm       = np.random.normal(np.mean(r["rpm"]),       (r["rpm"][1]-r["rpm"][0])/6,       n_pts)
        pressure  = np.random.normal(np.mean(r["pressure"]),  (r["pressure"][1]-r["pressure"][0])/6, n_pts)

        # apply age degradation
        vibration *= age_factor
        temp      *= (1 + row["age_years"] * 0.005)

        # inject fault if this machine is a "fault machine"
        will_fail  = (idx in fault_machines)
        fault_label = np.zeros(n_pts, dtype=int)

        if will_fail:
            fault_start = random.randint(int(n_pts * 0.5), int(n_pts * 0.85))
            intensity   = random.uniform(0.6, 1.4)
            temp      = add_fault_signature(temp,      fault_start, "temperature", intensity)
            vibration = add_fault_signature(vibration, fault_start, "vibration",   intensity)
            rpm       = add_fault_signature(rpm,       fault_start, "rpm",         intensity)
            pressure  = add_fault_signature(pressure,  fault_start, "pressure",    intensity)
            # label = 1 for readings within 120 h before fault
            pre_fault_start = max(0, fault_start - 120)
            fault_label[pre_fault_start:fault_start] = 1

        # clip to safe physical limits
        temp      = np.clip(temp,      20,  200)
        vibration = np.clip(vibration,  0,   20)
        rpm       = np.clip(rpm,        0, 3000)
        pressure  = np.clip(pressure,   0,   25)

        # add small random noise spikes
        noise_mask = np.random.random(n_pts) < 0.005
        temp[noise_mask]     += np.random.uniform(5, 15, noise_mask.sum())
        vibration[noise_mask]+= np.random.uniform(1,  5, noise_mask.sum())

        timestamps = [base_time + timedelta(hours=i) for i in range(n_pts)]

        for i in range(n_pts):
            all_records.append({
                "timestamp":   timestamps[i],
                "machine_id":  row["machine_id"],
                "temperature": round(float(temp[i]),      2),
                "vibration":   round(float(vibration[i]), 3),
                "rpm":         round(float(rpm[i]),        1),
                "pressure":    round(float(pressure[i]),  2),
                "fault_label": int(fault_label[i]),
            })

    df = pd.DataFrame(all_records)
    df.sort_values(["machine_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == "__main__":
    meta = generate_machine_metadata()
    meta.to_csv("machine_metadata.csv", index=False)
    print("machine_metadata.csv saved.")

    sensor_df = generate_sensor_data(meta)
    sensor_df.to_csv("sensor_data.csv", index=False)
    print(f"sensor_data.csv saved — {len(sensor_df):,} rows.")
    print(f"Fault readings: {sensor_df['fault_label'].sum():,}  "
          f"({sensor_df['fault_label'].mean()*100:.2f}%)")
