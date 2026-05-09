import serial
import serial.tools.list_ports
import time
import threading


def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        desc = (port.description or "").lower()
        hwid = (port.hwid or "").lower()
        if any(kw in desc or kw in hwid for kw in
               ["arduino", "ch340", "cp210", "usb serial", "acm"]):
            return port.device
    return None


class ServoController:
    def __init__(self, port=None, baud=9600, timeout=2.0):
        if port is None:
            port = find_arduino_port()
        if port is None:
            print("[ServoController] No Arduino found — servo disabled.")
            self._ser = None
            return
        try:
            self._ser = serial.Serial(port, baud, timeout=timeout)
            time.sleep(2.0)
            print(f"[ServoController] Connected on {port}")
        except serial.SerialException as e:
            print(f"[ServoController] Could not open {port}: {e} — servo disabled.")
            self._ser = None

    def trigger_death(self):
        if self._ser is None:
            return
        threading.Thread(target=self._send, daemon=True).start()

    def _send(self):
        try:
            self._ser.write(b'D')
            self._ser.flush()
            print("[ServoController] Death signal sent")
        except serial.SerialException as e:
            print(f"[ServoController] Send error: {e}")

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    def is_connected(self):
        return self._ser is not None and self._ser.is_open
