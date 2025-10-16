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

# --- FIX 1: Correct variable name from DATA_DIR to DATA_PATH ---
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

    index = 1  # Default index
    # Increment file index
    if not existed:
        index = 1
    else:
        # --- FIX 2: Correct typo from 'spilt' to 'split' ---
        last = os.path.basename(existed[-1]).split('.')[0]
        try:
            # --- FIX 3: Correct method from 'splitr' to 'rsplit' and use consistent variable 'index' ---
            last_index = int(last.rsplit('_', 1)[-1])
            index = last_index + 1
        except (ValueError, IndexError):
            index = 1
    
    # --- FIX 4: Use the correctly scoped 'index' variable ---
    return os.path.join(DATA_PATH, f"{DEFAULT}_{tag}_{index:03d}{FILETYPE}")

# Thread function for reading data from one port
def reader_thread(port: str):
    out_path = next_filename_for(port)
    print(f"[{port}] Writing to -> {out_path}")
    
    # Use 'with' for file handling to ensure it's closed properly
    try:
        record_coint=0   # <--- Add initialization here
        with open(out_path, "a", newline="") as f:
            # --- FIX 5: Correct variable name from 'write' to 'writer' for consistency ---
            writer = csv.writer(f)
            if f.tell() == 0:
                # --- FIX 6: Correct method name from 'writernow' to 'writerow' ---
                writer.writerow(["timestamp", "port", "payload"])  # CSV header

            ser = None
            while not stop_event.is_set():
                # Ensure serial connection
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
                    
                    # --- FIX 7: Correct decode error from 'igone' to 'ignore' ---
                    line = raw.decode(errors="ignore").strip()

                    # Only store lines containing CSI data keyword
                    # NOTE: Assuming the incoming typo is 'serian_num', corrected the 'if' condition
                    if "serian_num:" in line:
                        line = line.replace("serian_num", "serial_num")  # Typo fix
                        writer.writerow([datetime.now().isoformat(), port, line])
                        f.flush()
                        record_coint+=1   #####Count each entry as it is written.
                        if record_coint>=10000: ####### Check if the record count has reached 10000
                            print(f"[{port}] Reached 10,000 records. Stopping collection for this port.")
                            break
                except Exception as e:
                    print(f"[{port}] Read error: {e}; reconnecting in 2s...")
                    if ser and ser.is_open:
                        ser.close()
                    ser = None
                    time.sleep(2)

            # Cleanup on stop
            if ser and ser.is_open:
                ser.close()
                print(f"[{port}] Serial closed.")

    except IOError as e:
        print(f"[{port}] File error: {e}")
    
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
                    print("Stop event was set. Clearing to restart.")
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
                            # --- FIX 8: Correct typo from 'jon' to 'join' ---
                            t.join(timeout=5)  # Add a timeout for safety
                    print("All threads stopped.")
                else:
                    print("Threads are already stopping or stopped.")

            elif cmd == "e":
                if not stop_event.is_set():
                    print("Stopping all threads before exiting...")
                    stop_event.set()
                for p, t in threads.items():
                    if t and t.is_alive():
                        t.join(timeout=5) # Add a timeout for safety
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