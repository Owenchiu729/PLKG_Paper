#!/usr/bin/env python3
import socket
import time
import struct

# UDP broadcast address & port
BCAST_IP = "255.255.255.255"
BCAST_PORT = 9999

# Heartbeat frequency (10Hz = every 100ms)
INTERVAL = 0.1  # seconds

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    seq = 0
    print(f"📡 Heartbeat broadcasting every {INTERVAL}s → {BCAST_IP}:{BCAST_PORT}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            # pack seq into 4-byte unsigned int (network/big-endian)
            msg = struct.pack("!I", seq)
            sock.sendto(msg, (BCAST_IP, BCAST_PORT))
            seq += 1
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\n  Heartbeat stopped.")

if __name__ == "__main__":
    main()
