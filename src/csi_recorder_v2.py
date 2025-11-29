#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESP32 CSI Recorder (dual-port, CSV logger)

Features
- Auto-detect serial ports (CP210x/ttyUSB*/ttyACM*)
- Parse ESP32 CSI text lines: role=AP/STA ts=... rssi=... mac=... len=... csi=...
- Fixed-length CSI vector (first N subcarriers, padded with None)
- Periodic rate report (good/junk)
- Stop conditions:
    * --max N       : stop after N valid samples per port
    * --sec S       : stop after S seconds (all ports stop together)
    * --sync_stop   : broadcast stop when any/both port(s) reach(es) its target
        - --sync_mode any  (default): stop when ANY port hits its --max
        - --sync_mode both           : stop when ALL ports have >= --max
- Peer MAC filtering:
    * --peer_ap  <MAC>  : on STA role, only keep frames from AP MAC
    * --peer_sta <MAC>  : on AP role,  only keep frames from STA MAC

Typical usage
    python3 csi_recorder_v2.py --max 5000
    python3 csi_recorder_v2.py --sec 10 --sync_stop
    python3 csi_recorder_v2.py --max 5000 --sync_stop --sync_mode both \
        --peer_ap c0:49:ef:3f:f2:a5 --peer_sta b8:d6:1a:81:a7:78
"""

import os
import re
import csv
import glob
import time
import argparse
import threading
from pathlib import Path

import serial
import serial.tools.list_ports as list_ports

# ----------------------------- Constants -----------------------------
HOME_DIR = Path.home()
DATA_PATH = os.path.join(HOME_DIR, "PLKG_Project", "data", "collect")
DEFAULT_BASENAME = "csi"
CSV_EXT = ".csv"

REPORT_EVERY_SEC = 5.0
CSI_SUBCARRIERS = 128  # keep first 128 values (pad with None if fewer)
ESP32_PORT_HINTS = ["Silicon Labs", "CP210", "CH340", "USB Serial", "UART"]

# CSI line regex from firmware printing
CSI_LINE = re.compile(
    r"role=(?P<role>AP|STA)\s+ts=(?P<ts>\d+)\s+rssi=(?P<rssi>-?\d+)\s+mac=(?P<mac>[0-9a-f:]{17})\s+len=(?P<len>\d+)\s+csi=(?P<csi>.*)",
    re.IGNORECASE,
)

# Global stop coordination
stop_event = threading.Event()
COUNTS = {}               # per-port good counters (for sync_mode=both)
COUNTS_LOCK = threading.Lock()
# ---------------------------------------------------------------------

os.makedirs(DATA_PATH, exist_ok=True)


# ----------------------------- Helpers -----------------------------
def autodetect_ports():
    """Return a list of likely ESP32 serial device paths."""
    found = []
    for p in list_ports.comports():
        desc = f"{p.device} - {p.description}"
        if any(k in desc for k in ESP32_PORT_HINTS):
            found.append(p.device)
    if not found and os.name == "posix":
        # Fallback for Linux when description isn't helpful
        for p in list_ports.comports():
            if "ttyUSB" in p.device or "ttyACM" in p.device:
                found.append(p.device)
    return found


def next_csv_filename(port: str) -> str:
    """Create an incremented CSV filename based on port."""
    if os.name == "posix":
        tag = port.rsplit("/", 1)[-1].lower()
    else:
        tag = port.lower().replace(":", "")
    pattern = os.path.join(DATA_PATH, f"{DEFAULT_BASENAME}_{tag}_*{CSV_EXT}")
    existing = sorted(glob.glob(pattern))
    idx = 1
    if existing:
        last = os.path.basename(existing[-1]).split(".")[0]
        try:
            idx = int(last.rsplit("_", 1)[-1]) + 1
        except Exception:
            idx = 1
    return os.path.join(DATA_PATH, f"{DEFAULT_BASENAME}_{tag}_{idx:03d}{CSV_EXT}")


def open_serial(port: str, baud: int) -> serial.Serial:
    """Open a serial port with a short timeout for non-blocking reads."""
    return serial.Serial(port=port, baudrate=baud, timeout=1)


def parse_csi(line: str):
    """
    Parse one CSI text line.
    Returns tuple (role, ts_dev_us, rssi, mac, length, csi_list) or None.
    """
    m = CSI_LINE.search(line)
    if not m:
        return None
    role = m.group("role").upper()
    ts_dev_us = int(m.group("ts"))
    rssi = int(m.group("rssi"))
    mac = m.group("mac").lower()
    length = int(m.group("len"))

    raw = [x for x in m.group("csi").strip().split(",") if x]
    vals = []
    for x in raw:
        try:
            vals.append(int(x))
        except ValueError:
            # ignore non-integers inside the stream
            pass

    # keep first N subcarriers (pad to fixed length)
    if len(vals) > CSI_SUBCARRIERS:
        vals = vals[:CSI_SUBCARRIERS]
    if len(vals) < CSI_SUBCARRIERS:
        vals += [None] * (CSI_SUBCARRIERS - len(vals))

    return role, ts_dev_us, rssi, mac, length, vals
# ---------------------------------------------------------------------


# ----------------------------- Worker -----------------------------
def reader_thread(
    port: str,
    baud: int,
    max_records: int | None,
    peer_ap: str | None,
    peer_sta: str | None,
):
    """
    One reader per port.
    - peer_ap  : STA role keeps only frames with mac==peer_ap
    - peer_sta : AP  role keeps only frames with mac==peer_sta
    """
    out_path = next_csv_filename(port)
    print(f"[{port}] Writing -> {out_path}")

    good = junk = 0
    window_good = 0
    last_report = time.time()
    ser = None

    # Normalize peer MAC strings to lowercase
    if peer_ap:
        peer_ap = peer_ap.lower()
    if peer_sta:
        peer_sta = peer_sta.lower()

    header = ["ts_host_ns", "role", "ts_dev_us", "rssi", "mac", "len"] + [
        f"sc{i+1}" for i in range(CSI_SUBCARRIERS)
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        while not stop_event.is_set():
            # Ensure port is open
            if ser is None or not ser.is_open:
                try:
                    ser = open_serial(port, baud)
                    print(f"[{port}] Serial connected @ {baud}")
                except Exception as e:
                    print(f"[{port}] open failed: {e}; retry in 2s")
                    time.sleep(2)
                    continue

            try:
                raw = ser.readline()
                if raw:
                    ts_host_ns = time.monotonic_ns()
                    line = raw.decode(errors="ignore").strip()

                    parsed = parse_csi(line)
                    if not parsed:
                        junk += 1
                    else:
                        role, ts_dev_us, rssi, mac, length, vals = parsed

                        # --- Peer MAC filter (keep only frames from the other device) ---
                        # STA role should keep frames from AP; AP role keeps frames from STA.
                        if role == "STA" and peer_ap and mac != peer_ap:
                            junk += 1
                            continue
                        if role == "AP" and peer_sta and mac != peer_sta:
                            junk += 1
                            continue
                        # -----------------------------------------------------------------

                        writer.writerow([ts_host_ns, role, ts_dev_us, rssi, mac, length] + vals)
                        good += 1
                        window_good += 1

                        # update shared counter for sync_mode=both
                        with COUNTS_LOCK:
                            COUNTS[port] = good

                        # Flush periodically to avoid data loss
                        if good % 500 == 0:
                            f.flush()

                        # Per-port --max stop (for simple mode or sync_mode=any)
                        if max_records is not None and good >= max_records:
                            print(f"[{port}] reached {max_records} records.")
                            # For sync_stop=any, we stop everyone right now.
                            # For sync_stop=both, the main thread watcher will call stop_event.set()
                            break

                # Periodic rate report
                now = time.time()
                if now - last_report >= REPORT_EVERY_SEC:
                    rate = window_good / max(1e-9, (now - last_report))
                    print(f"[{port}] rate ~ {rate:.1f} samples/s | good={good} junk={junk}")
                    window_good = 0
                    last_report = now

            except Exception as e:
                print(f"[{port}] read error: {e}; reconnecting in 2s")
                try:
                    if ser and ser.is_open:
                        ser.close()
                except Exception:
                    pass
                ser = None
                time.sleep(2)

    # Final close
    try:
        if ser and ser.is_open:
            ser.close()
    except Exception:
        pass

    print(f"[{port}] done. good={good} junk={junk}")
# ---------------------------------------------------------------------


# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="ESP32 CSI dual-port CSV recorder")
    ap.add_argument("--port", action="append",
                    help="Repeatable. If omitted, auto-detect (e.g., --port /dev/ttyUSB0 --port /dev/ttyUSB1)")
    ap.add_argument("--baud", type=int, default=115200, help="Serial baudrate (default: 115200)")

    # Stop conditions
    ap.add_argument("--max", type=int, default=None, help="Stop after N valid records per port")
    ap.add_argument("--sec", type=int, default=None, help="Stop after S seconds (all ports)")

    # Sync behavior
    ap.add_argument("--sync_stop", action="store_true",
                    help="Coordinate stopping across ports")
    ap.add_argument("--sync_mode", choices=["any", "both"], default="any",
                    help="When used with --sync_stop and --max:\n"
                         "  any  : stop when ANY port reaches --max (default)\n"
                         "  both : stop when ALL ports have >= --max")

    # Peer MAC filtering (highly recommended for clean AP<->STA datasets)
    ap.add_argument("--peer_ap",  type=str, default=None,
                    help="AP device MAC (STA role will keep only frames from this MAC)")
    ap.add_argument("--peer_sta", type=str, default=None,
                    help="STA device MAC (AP role will keep only frames from this MAC)")

    args = ap.parse_args()

    ports = args.port or autodetect_ports()
    if not ports:
        print("No serial ports found. Use --port to specify manually.")
        return

    print(f"Ports: {ports} | Baud={args.baud} | Max={args.max} | Sec={args.sec} | SyncStop={args.sync_stop}")

    # If --sec is provided, schedule a timer to stop everything after S seconds
    if args.sec is not None:
        def _timer():
            time.sleep(max(0, args.sec))
            print(f"\n[Timer] {args.sec}s reached, stopping...")
            stop_event.set()
        threading.Thread(target=_timer, daemon=True).start()

    # Initialize counts for sync_mode=both watcher
    with COUNTS_LOCK:
        for p in ports:
            COUNTS[p] = 0

    # Start readers
    threads = []
    for p in ports:
        t = threading.Thread(
            target=reader_thread,
            args=(p, args.baud, args.max, args.peer_ap, args.peer_sta),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # If requested, coordinate stopping across ports
    # 1) sync_mode=any : if ANY port reaches --max, stop all immediately
    #    (this is implemented by watching COUNTS and comparing against --max)
    # 2) sync_mode=both: wait until ALL ports reach --max, then stop all
    if args.sync_stop and args.max is not None:
        def watcher():
            mode = args.sync_mode
            target = args.max
            print(f"[SyncWatcher] mode={mode}, target={target}")
            while not stop_event.is_set():
                with COUNTS_LOCK:
                    counts = {k: COUNTS.get(k, 0) for k in ports}
                if mode == "any":
                    if any(v >= target for v in counts.values()):
                        print(f"[SyncWatcher] condition met (any). counts={counts}")
                        stop_event.set()
                        break
                else:  # both
                    if all(v >= target for v in counts.values()):
                        print(f"[SyncWatcher] condition met (both). counts={counts}")
                        stop_event.set()
                        break
                time.sleep(0.05)
        threading.Thread(target=watcher, daemon=True).start()

    # Block until all readers finish
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
