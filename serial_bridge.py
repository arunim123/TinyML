import json
import logging
import queue
import threading
import time

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SerialBridge")

def list_available_serial_ports():
    """Return a list of available serial port names on the system."""
    if not SERIAL_AVAILABLE:
        return []
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

class ESP32SerialBridge:
    def __init__(self, port=None, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        self.latest_telemetry = {
            "count": 0,
            "pred": 0,
            "conf": 0.95,
            "mag": 1.0,
            "sparsity": 0.05,
            "buffer": 0,
            "static_ram_kb": 33.6,
            "latency_ms": 1.8,
            "connected": False
        }
        
        self.history_queue = queue.Queue(maxsize=100)

    def connect(self, port=None):
        if port:
            self.port = port
        if not self.port or not SERIAL_AVAILABLE:
            logger.warning("Serial library unavailable or no port specified.")
            return False

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1.0)
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            logger.info(f"Connected to ESP32 on serial port {self.port} at {self.baudrate} baud.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to serial port {self.port}: {e}")
            return False

    def disconnect(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        logger.info("Serial connection closed.")

    def _read_loop(self):
        while self.running and self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                
                if line.startswith("{") and line.endswith("}"):
                    try:
                        data = json.loads(line)
                        if data.get("type") == "telemetry":
                            with self.lock:
                                data["connected"] = True
                                self.latest_telemetry.update(data)
                                if self.history_queue.full():
                                    self.history_queue.get_nowait()
                                self.history_queue.put(data)
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                logger.error(f"Serial read error: {e}")
                time.sleep(0.1)

    def send_label(self, label: int):
        """Send label correction command back to ESP32 for on-device adaptation."""
        if self.ser and self.ser.is_open:
            cmd = json.dumps({"cmd": "label", "val": label}) + "\n"
            try:
                self.ser.write(cmd.encode('utf-8'))
                logger.info(f"Sent adaptation label command to ESP32: {cmd.strip()}")
                return True
            except Exception as e:
                logger.error(f"Failed to send label command over serial: {e}")
                return False
        return False

    def get_latest_telemetry(self):
        with self.lock:
            return dict(self.latest_telemetry)
