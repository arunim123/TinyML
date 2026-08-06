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
        self.serial_logs = []
        self.last_received_time = 0.0
        self.reconnect_delay = 2.0

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
            # Still start read loop thread to allow automatic reconnection once available
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            return False

    def disconnect(self):
        self.running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        logger.info("Serial connection closed.")

    def _read_loop(self):
        self.last_received_time = time.time()
        while self.running:
            # Check connection timeout (3.0 seconds)
            if time.time() - self.last_received_time > 3.0:
                with self.lock:
                    self.latest_telemetry["connected"] = False

            if not self.ser or not self.ser.is_open:
                try:
                    self.ser = serial.Serial(self.port, self.baudrate, timeout=1.0)
                    self.last_received_time = time.time()
                    with self.lock:
                        self.latest_telemetry["connected"] = True
                    logger.info(f"Reconnected to ESP32 on serial port {self.port}.")
                except Exception:
                    time.sleep(self.reconnect_delay)
                    continue

            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                
                # Telemetry heartbeat updated
                self.last_received_time = time.time()
                with self.lock:
                    self.latest_telemetry["connected"] = True

                # Generate host-level timestamp with millisecond precision: [HH:MM:SS.mmm]
                current_time = time.time()
                milli = int((current_time % 1) * 1000)
                timestamp_str = time.strftime("%H:%M:%S", time.localtime(current_time)) + f".{milli:03d}"
                
                # Format raw message as received
                formatted_log = f"[{timestamp_str}] {line}"
                with self.lock:
                    self.serial_logs.append(formatted_log)
                    if len(self.serial_logs) > 1000:
                        self.serial_logs.pop(0)

                # Parse JSON telemetry packet
                if line.startswith("{") and line.endswith("}"):
                    try:
                        data = json.loads(line)
                        if data.get("type") == "telemetry":
                            with self.lock:
                                self.latest_telemetry.update(data)
                                self.latest_telemetry["connected"] = True
                            if self.history_queue.full():
                                self.history_queue.get_nowait()
                            self.history_queue.put(data)
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                logger.error(f"Serial read error on port {self.port}: {e}")
                with self.lock:
                    self.latest_telemetry["connected"] = False
                if self.ser:
                    try:
                        self.ser.close()
                    except Exception:
                        pass
                    self.ser = None
                time.sleep(self.reconnect_delay)

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

