import os
import csv
import glob
import time
import threading
from datetime import datetime
import serial

#==============Config==========================================
PORTS     = ['/dev/ttyUSB0', '/dev/ttyUSB1']   # Two ESP32 devices
BAUDRATE  = 115200                             # Serial baud rate
DATA_PATH = "/home/owen/PLKG_Project/data/collect"   # Folder for saving files
DEFAULT   = "csi_data"                         # File prefix
FILETYPE  = ".csv"                             # File extension
#==============================================================

# mkdir folder
os.makedirs(DATA_PATH, exist_ok=True)
stop_event = threading.Event()

def connect_serial(port, baudrate):
    return serial.Serial(port, baudrate, timeout=1)

# Generate next CSV filename for each port
def next_filename_for(port: str) -> str:
    tail = port.rsplit('/', 1)[-1].lower()
    tag = tail.replace('tty', '')
    pattern = os.path.join(DATA_PATH, f"{DEFAULT}_{tag}_*{FILETYPE}")
    existed = sorted(glob.glob(pattern))
    index = 1
    if existed:
        last = os.path.basename(existed[-1]).split('.')[0]
        try:
            last_index = int(last.rsplit('_', 1)[-1])
            index = last_index + 1
        except (ValueError, IndexError):
            index = 1
    return os.path.join(DATA_PATH, f"{DEFAULT}_{tag}_{index:03d}{FILETYPE}")

# Thread function for reading data from one port
def reader_thread(port: str):
    out_path = next_filename_for(port)
    print(f"[{port}] Writing to -> {out_path}")
    
    record_coint = 0
    ser = None
    
    try:
        # 'with' statement manages the file, ensuring it's closed properly on exit
        with open(out_path, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["timestamp", "port", "payload"])

            # The main loop MUST be inside the 'with' block
            while not stop_event.is_set():
                if ser is None or not ser.is_open:
                    try:
                        ser = connect_serial(port, BAUDRATE)
                        print(f"[{port}] Serial connected.")
                    except Exception as e:
                        print(f"[{port}] Failed to open: {e}; retrying in 2s...")
                        time.sleep(2)
                        continue
                
                try:
                    raw = ser.readline()
                    if not raw:
                        continue
                    
                    line = raw.decode(errors="ignore").strip()

                    # --- Corrected logic: Directly check for the correct keyword ---
                    if "serial_num:" in line:
                        writer.writerow([datetime.now().isoformat(), port, line])
                        f.flush()
                        record_coint += 1
                        if record_coint >= 10000:
                            print(f"[{port}] Reached 10,000 records. Stopping collection for this port.")
                            break
                except Exception as e:
                    print(f"[{port}] Read error: {e}; reconnecting in 2s...")
                    if ser and ser.is_open:
                        ser.close()
                    ser = None
                    time.sleep(2)

    except IOError as e:
        print(f"[{port}] File error: {e}")
    finally:
        # Cleanup: Ensure serial port is closed when the thread stops for any reason
        if ser and ser.is_open:
            ser.close()
            print(f"[{port}] Serial closed.")
        print(f"[{port}] Stopped collecting.")


##### Main control loop
def main():
    print("Commands: r = start, q = stop, e = exit")
    threads = {p: None for p in PORTS}
    
    try:
        while True:
            cmd = input(">> ").strip().lower()
            if cmd == "r":
                if stop_event.is_set():
                    stop_event.clear()

                for p in PORTS:
                    if threads[p] is None or not threads[p].is_alive():
                        t = threading.Thread(target=reader_thread, args=(p,), daemon=True)
                        t.start()
                        threads[p] = t
                print("Started collecting from all ESP32 devices.")

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