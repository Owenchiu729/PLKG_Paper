import os
import csv
import glob
import time
import threading
import argparse
from datetime import datetime
import serial
import serial.tools.list_ports as list_ports
from pathlib import Path

# ==================== Config ====================
HOME_DIR = Path.home()  # /home/owen or C:\Users\owen
DATA_PATH = os.path.join(HOME_DIR, "PLKG_Project", "data", "collect")
DEFAULT   = "csi_data"
FILETYPE  = ".csv"
REPORT_INTERVAL_SEC = 5       # 每個埠每 5 秒印出一次有效速率
JUNK_LOG_INTERVAL   = 500     # 每 500 行垃圾訊息印一次統計
# =================================================

ESP32_KEYWORDS = ["Silicon Labs", "CP210x", "CH340", "USB Serial", "UART"]

os.makedirs(DATA_PATH, exist_ok=True)
stop_event = threading.Event()

# 讓每個 reader 執行緒可以即時查詢統計（給 command 's' 用）
_port_stats_lock = threading.Lock()
_port_stats = {}  # {port: {"good": int, "junk": int, "last_rate": float, "started": float, "outfile": str}}


def connect_serial(port, baudrate):
    return serial.Serial(port, baudrate, timeout=1)


def auto_detect_ports():
    ports = list(list_ports.comports())
    detected = []
    if not ports:
        return []
    for p in ports:
        desc = f"{p.device} - {p.description}"
        if any(k in desc for k in ESP32_KEYWORDS):
            detected.append(p.device)
    # fallback: Linux ttyUSB/ttyACM 全撈
    if not detected and os.name == 'posix':
        for p in ports:
            if "ttyUSB" in p.device or "ttyACM" in p.device:
                if p.device not in detected:
                    detected.append(p.device)
    return detected


def next_filename_for(port: str, out_path: str) -> str:
    # /dev/ttyUSB0 -> usb0； COM3 -> com3
    if os.name == 'posix':
        tail = port.rsplit('/', 1)[-1].lower()
        tag = tail.replace('tty', '')
    else:
        tag = port.lower()

    pattern = os.path.join(out_path, f"{DEFAULT}_{tag}_*{FILETYPE}")
    existed = sorted(glob.glob(pattern))
    index = 1
    if existed:
        last = os.path.basename(existed[-1]).split('.')[0]
        try:
            last_index = int(last.rsplit('_', 1)[-1])
            index = last_index + 1
        except (ValueError, IndexError):
            index = 1
    return os.path.join(out_path, f"{DEFAULT}_{tag}_{index:03d}{FILETYPE}")


def _update_stats(port: str, **kwargs):
    with _port_stats_lock:
        st = _port_stats.setdefault(port, {"good": 0, "junk": 0, "last_rate": 0.0, "started": time.time(), "outfile": ""})
        st.update({k: v for k, v in kwargs.items() if k in st or True})


# Thread function for reading data from one port

def reader_thread(port: str, baudrate: int, out_path: str, max_records: int, filter_key: str):
    out_file = next_filename_for(port, out_path)
    print(f"[{port}] Writing to -> {out_file}")

    record_count = 0
    junk_cnt = 0
    good_cnt = 0
    ser = None

    _update_stats(port, good=0, junk=0, last_rate=0.0, started=time.time(), outfile=out_file)

    # 速率統計
    last_report_t = time.time()
    window_good = 0

    try:
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ts_ns", "port", "payload"])  # 保持現有 schema

            while not stop_event.is_set():
                # 1) 確保序列連線
                if ser is None or not ser.is_open:
                    try:
                        ser = connect_serial(port, baudrate)
                        print(f"[{port}] Serial connected @ {baudrate}.")
                    except Exception as e:
                        print(f"[{port}] Failed to open: {e}; retrying in 2s...")
                        time.sleep(2)
                        continue

                # 2) 讀取
                try:
                    raw = ser.readline()
                    if not raw:
                        # timeout，直接下一輪
                        pass
                    else:
                        ts = time.monotonic_ns()
                        line = raw.decode(errors="ignore").strip()

                        if filter_key in line:
                            # 真正的 CSI 行
                            writer.writerow([ts, port, line])
                            record_count += 1
                            good_cnt += 1
                            window_good += 1

                            # 定期 flush，避免資料遺失
                            if record_count % 500 == 0:
                                f.flush()

                            if record_count >= max_records:
                                print(f"[{port}] Reached {max_records} records. Stopping.")
                                break
                        else:
                            # 垃圍訊息（例如 promt/cb/udp_server_task/wifi_softap 等）
                            junk_cnt += 1
                            if junk_cnt % JUNK_LOG_INTERVAL == 0:
                                print(f"[{port}] filtered junk lines: {junk_cnt}")

                    # 3) 速率報告（每 REPORT_INTERVAL_SEC 秒）
                    now = time.time()
                    if now - last_report_t >= REPORT_INTERVAL_SEC:
                        rate = window_good / (now - last_report_t) if (now - last_report_t) > 0 else 0.0
                        print(f"[{port}] effective CSI rate ~ {rate:.1f} samples/s | good={good_cnt} junk={junk_cnt}")
                        _update_stats(port, good=good_cnt, junk=junk_cnt, last_rate=rate)
                        window_good = 0
                        last_report_t = now

                except Exception as e:
                    print(f"[{port}] Read error: {e}; reconnecting in 2s...")
                    if ser and ser.is_open:
                        ser.close()
                    ser = None
                    time.sleep(2)

    except IOError as e:
        print(f"[{port}] File error: {e}")
    finally:
        # 收尾
        if ser and ser.is_open:
            ser.close()
            print(f"[{port}] Serial closed.")
        print(f"[{port}] Stopped collecting. Final: good={good_cnt}, junk={junk_cnt}")
        _update_stats(port, good=good_cnt, junk=junk_cnt)


def _print_stats_table():
    # 簡易統計表（命令 's' 顯示）
    with _port_stats_lock:
        if not _port_stats:
            print("(no active ports)")
            return
        print("\nPORT STATS:\nport\tgood\tjunk\trate(sps)\toufile")
        for p, st in _port_stats.items():
            print(f"{p}\t{st.get('good',0)}\t{st.get('junk',0)}\t{st.get('last_rate',0):.1f}\t{st.get('outfile','')}")
        print()


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Multi-port CSI collector (with junk filter & rate monitor)")
    parser.add_argument("--port", action="append", help="指定要監聽的 Serial Port (可多次使用)")
    parser.add_argument("--baud", type=int, default=115200, help="Baudrate (預設: 115200)")
    parser.add_argument("--path", default=DATA_PATH, help=f"輸出資料夾 (預設: {DATA_PATH})")
    parser.add_argument("--max", type=int, default=50000, help="每個 Port 收集的筆數上限 (預設: 50000)")
    parser.add_argument("--filter", default="serial_num:", help="只儲存含此關鍵字的行 (預設: 'serial_num:')")

    args = parser.parse_args()

    # 決定 ports
    ports_to_use = args.port
    if not ports_to_use:
        print("No --port specified, auto-detecting...")
        ports_to_use = auto_detect_ports()
        if not ports_to_use:
            print("❌ 找不到任何 Serial Port，請接上 ESP32 或用 --port 指定")
            return
        print(f"✅ Auto-detected ports: {ports_to_use}")
    else:
        print(f"✅ Using specified ports: {ports_to_use}")

    print("\n--- CSI Collector ---")
    print(f"Baud: {args.baud} | Max Records: {args.max} | Filter: '{args.filter}'")
    print("Commands: r = start, q = stop, s = stats, e = exit")

    threads = {p: None for p in ports_to_use}

    try:
        while True:
            cmd = input(">> ").strip().lower()

            if cmd == "r":
                if stop_event.is_set():
                    stop_event.clear()
                for p in ports_to_use:
                    if threads[p] is None or not threads[p].is_alive():
                        print(f"Starting thread for {p}...")
                        t = threading.Thread(target=reader_thread, args=(p, args.baud, args.path, args.max, args.filter))
                        t.start()
                        threads[p] = t
                print("Started collecting from all devices.")

            elif cmd == "q":
                if not stop_event.is_set():
                    print("Stopping all threads...")
                    stop_event.set()
                    for p, t in threads.items():
                        if t and t.is_alive():
                            t.join(timeout=5)
                    print("All threads stopped.")
                else:
                    print("Threads are already stopping or stopped.")

            elif cmd == "s":
                _print_stats_table()

            elif cmd == "e":
                if not stop_event.is_set():
                    print("Stopping all threads before exiting...")
                    stop_event.set()
                for p, t in threads.items():
                    if t and t.is_alive():
                        t.join(timeout=5)
                print("Exiting program.")
                break

            else:
                print("Unknown command. Use: r = start, q = stop, s = stats, e = exit")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting...")
        stop_event.set()
        for t in threads.values():
            if t and t.is_alive():
                t.join(timeout=5)


if __name__ == "__main__":
    main()
