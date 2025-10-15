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
DATA_PATH = "./collect_data"                   # Folder for saving files
DEFAULT   = "csi_data"                         # File prefix
FILETYPE  = ".csv"                             # File extension
#==============================================================

#mkdir folder 
os.makedirs(DATA_DIR, exist_ok=True)
stop_event = threading.Event()
def connect_serial(port , baudrate):
    return serial.Serial(port, baudrate, timeout=1)
# Generate next CSV filename for each port
def next_filename_for(port:str)-> str:
    tail = port.rsplit('/',1)[-1].lower()
    tag = tail.replace('tty','')
    pattern =os.path.join(DATA_PATH, f"{DEFAULT}_{tag}_*{FILETYPE}")
    existed = sorted(glob.glob(pattern))


#Increment file index
if not existed:
    index=1

else:
    last=os.path.basename(existed[-1]).spilt('.')[0]
    try:
        idx= int(last.splitr('_'))[-1]+1
    except ValueError:
        index=1
 return os.path.join(DATA_PATH, f"{DEFAULT}_{tag}_{idx:03d}{FILETYPE}"  )     



# Thread function for reading data from one port
def reader_thread (port:str):
out_path = next_filename_for(port)
print(f"[{port}] Writing to -> {out_path}")
with open(out_path,"a",newline="")as f:
    write = csv.writer(f)
    if f.tell() ==0:
        writer.writernow(["timestamp", "port", "payload"])  # CSV header

ser = None
while not stop_event.is_set():
   # Ensure serial connection








def connect_serial(port, baudrate): 
    timeout = 1  # set timeout

    # open serial
    ser = serial.Serial(port, baudrate, timeout = timeout)

    return ser

def generate_filename():
    global Data_Path

    path = Data_Path + r"/" + Defult + r"*"+ filetype
    filelist = glob.glob(path)
    if len(filelist) == 0:
        index = 1
    else:
        index = 1
        for s in filelist:
            temp = int(s[len(Data_Path)+len(Defult)+1:-len(filetype)]) + 1
            index = max(temp,index)
    output_file =  Data_Path + r"/" + Defult + str(index) + filetype

    return output_file

def read_csi(ser):
    file = generate_filename()

    with open(file, 'a') as f:
        print(f"Start Collecting...")
        print(f"Writing CSI data in", file)
        try:
            while not stop_event.is_set():
                line = ser.readline().decode().strip()
                if "serial_num:" in line:
                    line = str(datetime.now()) + " ,"+ line.replace("serian_num","").strip()
                    f.write(line + '\n\n')
                    
        except KeyboardInterrupt:
            print(f"Stop Collecting...")
            f.close()
    f.close()

def main():
    ser = connect_serial(port, baudrate)
    task = None

    while True:
        t = input("\n>>")
        if t == "r" and (task is None or not task.is_alive()):
            stop_event.clear()
            task = threading.Thread(target=read_csi, args=(ser,))
            task.start()
        elif t == "q" and task is not None:
            stop_event.set()
            task.join()
            print(f"Stop Collecting...")
            #print(f"Serial closed...")
        elif t == 'e':
            ser.close()
            break
        else:
            pass
        time.sleep(1)

if __name__ == "__main__":
    main()
