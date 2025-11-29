# -*- coding: utf-8 -*-
import pandas as pd
import sys

def count_csi_rate(file):
    # Load CSV
    df = pd.read_csv(file)

    # Filter valid timestamp rows (ts_ns > 0)
    df = df[df['ts_ns'] > 0]

    # Keep only CSI rows that include "serial_num"
    df = df[df['payload'].str.contains('serial_num:', na=False)]

    # If no valid rows found, return None
    if len(df) == 0:
        return None, None, None

    # Convert nanoseconds to seconds
    df['ts_s'] = df['ts_ns'] / 1e9

    # Compute time duration
    duration = df['ts_s'].iloc[-1] - df['ts_s'].iloc[0]

    # Count CSI samples
    samples = len(df)

    # Samples per second
    rate = samples / duration if duration > 0 else 0

    return duration, samples, rate


if __name__ == "__main__":
    # Require two input files: AP and STA CSV
    if len(sys.argv) < 3:
        print("Usage: python csi_rate_test.py ap.csv sta.csv")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    dur1, samp1, rate1 = count_csi_rate(file1)
    dur2, samp2, rate2 = count_csi_rate(file2)

    print("=== CSI Sample Rate Test ===")

    if rate1 is not None:
        print(f"{file1}:")
        print(f"  Duration: {dur1:.2f} sec")
        print(f"  Samples:  {samp1}")
        print(f"  Rate:     {rate1:.2f} samples/sec")
    else:
        print(f"{file1}: No valid CSI samples")

    if rate2 is not None:
        print(f"{file2}:")
        print(f"  Duration: {dur2:.2f} sec")
        print(f"  Samples:  {samp2}")
        print(f"  Rate:     {rate2:.2f} samples/sec")
    else:
        print(f"{file2}: No valid CSI samples")

    if rate1 and rate2:
        avg = (rate1 + rate2) / 2
        print(f"\nAverage CSI Rate: {avg:.2f} samples/sec")
