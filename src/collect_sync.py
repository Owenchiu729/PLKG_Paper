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

#==============Config==========================================
# 動態建立跨平台的 "家目錄" 路徑
HOME_DIR = Path.home()  # 自動取得 /home/owen (Linux/Mac) 或 C:\Users\owen (Windows)
DATA_PATH = os.path.join(HOME_DIR, "PLKG_Project", "data", "collect") # <--- 使用 os.path.join 來組合
DEFAULT   = "csi_data"                         # 檔名前綴
FILETYPE  = ".csv"                             # 副檔名
#==============================================================


# 自動偵測關鍵字
ESP32_KEYWORDS = ["Silicon Labs", "CP210x", "CH340", "USB Serial", "UART"]

# mkdir folder
os.makedirs(DATA_PATH, exist_ok=True)
stop_event = threading.Event()

def connect_serial(port, baudrate):
    return serial.Serial(port, baudrate, timeout=1)
def auto_detect_ports():
#自動偵測所有可能的 ESP32 埠 
    ports = list(list_ports.comports())
    detected = []
    if not ports:
        return []
    for p in  ports:
        desc = f"{p.device} - {p.description}"
        if any(k in desc for k in ESP32_KEYWORDS):
                detected.append(p.device)
# 如果沒偵測到關鍵字，退一步，返回所有 ttyUSB 和 ttyACM (Linux)
    if not detected and os.name == 'posix':
        for p in ports:
            if "ttyUSB" in p.device or "ttyACM" in p.device:
                 if p.device not in detected:
                     detected.append(p.device)
    return detected


def next_filename_for(port: str, out_path: str) -> str:
    # 從 /dev/ttyUSB0 -> usb0 或 COM3 -> com3
    if os.name == 'posix':
        tail = port.rsplit('/', 1)[-1].lower()
        tag = tail.replace('tty', '')
    else:
        tag = port.lower()
    
    # 這裡的 out_path 都是小寫
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

# Thread function for reading data from one port
def reader_thread(port: str, baudrate: int, out_path: str, max_records: int, filter_key: str):
#執行緒函式，用於從單一序列埠讀取資料並寫入檔案。
    
    out_file = next_filename_for(port, out_path)
    print(f"[{port}] Writing to -> {out_file}")
    
    record_count = 0
    ser = None
    try:
        # vvvv 【錯誤修正】 vvvv
        # 這裡的 f 後面必須有冒號 ':'
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            # 使用 'ts_ns' (nanoseconds) 作為表頭
            writer.writerow(["ts_ns", "port", "payload"])

            while not stop_event.is_set():
                # 1. 檢查並建立序列埠連線
                if ser is None or not ser.is_open:
                    try:
                        ser = connect_serial(port, baudrate)
                        print(f"[{port}] Serial connected @ {baudrate}.")
                    except Exception as e:
                        print(f"[{port}] Failed to open: {e}; retrying in 2s...")
                        time.sleep(2)
                        continue
                
                # 2. 讀取與寫入資料
                try:
                    raw = ser.readline()
                    if not raw:
                        continue # Timeout, 繼續迴圈
                    
                    # 在讀到資料 *後* 立刻取得單調時間戳
                    ts = time.monotonic_ns()
                    line = raw.decode(errors="ignore").strip()

                    # 依據關鍵字過濾
                    if filter_key in line:
                        # 寫入 'ts_ns'
                        writer.writerow([ts, port, line])
                        record_count += 1
                        
                        # 每 500 筆 flush 一次，避免 I/O 過於頻繁
                        if record_count % 500 == 0:
                            f.flush()

                        if record_count >= max_records:
                            print(f"[{port}] Reached {max_records} records. Stopping.")
                            break # 達到上限，跳出 while 迴圈
                            
                except Exception as e:
                    print(f"[{port}] Read error: {e}; reconnecting in 2s...")
                    if ser and ser.is_open:
                        ser.close()
                    ser = None
                    time.sleep(2)

    except IOError as e:
        print(f"[{port}] File error: {e}")
    finally:
        # 執行緒收尾：確保序列埠被關閉
        if ser and ser.is_open:
            ser.close()
            print(f"[{port}] Serial closed.")
        print(f"[{port}] Stopped collecting.")

   


##### Main control loop
def main():
    parser = argparse.ArgumentParser(description="Multi-port CSI collector")
    parser.add_argument("--port", 
                        action="append", 
                        help="指定要監聽的 Serial Port (可多次使用, e.g., --port /dev/ttyUSB0 --port /dev/ttyUSB1)")
    parser.add_argument("--baud", type=int, default=115200, help="Baudrate (預設: 115200)")
    
    # 這裡的 --path 都是小寫
    parser.add_argument("--path", default=DATA_PATH, help=f"輸出資料夾 (預設: {DATA_PATH})")
    parser.add_argument("--max", 
                        type=int, 
                        default=50000,
                        help="每個 Port 收集的筆數上限 (預設: 50000)")
    parser.add_argument("--filter", default="serial_num:", help="要儲存的資料行必須包含的關鍵字 (預設: 'serial_num:')")
    
    args = parser.parse_args()
    
    # 決定要監聽哪些 ports
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
    print("Commands: r = start, q = stop, e = exit")
    
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
                        # 移除 daemon=True，確保安全關閉
                        
                        # 這裡的 args.path 都是小寫
                        t = threading.Thread(target=reader_thread, 
                                             args=(p, args.baud, args.path, args.max, args.filter))
                        t.start()
                        threads[p] = t
                print("Started collecting from all devices.")

            elif cmd == "q":
                if not stop_event.is_set():
                    print("Stopping all threads...")
                    stop_event.set()
                    for p, t in threads.items():
                        if t and t.is_alive():
                            t.join(timeout=5) # 等待 5 秒
                    print("All threads stopped.")
                else:
                    print("Threads are already stopping or stopped.")

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
                print("Unknown command. Use: r = start, q = stop, e = exit")
                
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting...")
        stop_event.set()
        for t in threads.values():
            if t and t.is_alive():
                t.join(timeout=5)

if __name__ == "__main__":
    main()