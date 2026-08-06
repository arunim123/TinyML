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

        # Hardware Profile parameters
        self.board_name = "Unknown Device"
        self.chip_family = "Unavailable"
        self.firmware_version = "Unavailable"
        self.build_version = "Unavailable"
        self.app_version = "Unavailable"
        self.sdk_version = "Unavailable"
        self.uptime = "Unavailable"
        
        self.usb_board_fallback = "Unknown Device"
        self.usb_chip_fallback = "Unavailable"
        self.connection_start_time = 0.0

    def connect(self, port=None):
        if port:
            self.port = port
        if not self.port or not SERIAL_AVAILABLE:
            logger.warning("Serial library unavailable or no port specified.")
            return False

        # Reset connection telemetry & settings
        self.board_name = "Unknown Device"
        self.chip_family = "Unavailable"
        self.firmware_version = "Unavailable"
        self.build_version = "Unavailable"
        self.app_version = "Unavailable"
        self.sdk_version = "Unavailable"
        self.uptime = "Unavailable"
        
        self.usb_board_fallback = "Unknown Device"
        self.usb_chip_fallback = "Unavailable"
        self.connection_start_time = time.time()

        # Fallback USB metadata scan
        try:
            ports = list(serial.tools.list_ports.comports())
            for p in ports:
                if p.device == self.port:
                    desc = p.description.lower() if p.description else ""
                    vid = p.vid if p.vid else 0
                    pid = p.pid if p.pid else 0
                    
                    if vid == 0x303a:
                        self.usb_chip_fallback = "ESP32"
                        if pid == 0x1001:
                            self.usb_board_fallback = "ESP32-S3 DevKitC-1"
                            self.usb_chip_fallback = "ESP32-S3"
                        elif pid in [0x1002, 0x8004]:
                            self.usb_board_fallback = "ESP32-C3 SuperMini"
                            self.usb_chip_fallback = "ESP32-C3"
                        elif pid == 0x1003:
                            self.usb_board_fallback = "ESP32-C6"
                            self.usb_chip_fallback = "ESP32-C6"
                        else:
                            self.usb_board_fallback = "ESP32 DevKit"
                    elif vid == 0x10c4 and pid == 0xea60:
                        self.usb_board_fallback = "ESP32 DevKit V1"
                        self.usb_chip_fallback = "ESP32-WROOM-32"
                    elif vid == 0x1a86 and pid == 0x7523:
                        self.usb_board_fallback = "ESP32 DevKit V1 (CH340)"
                        self.usb_chip_fallback = "ESP32"
                    elif "esp32-s3" in desc:
                        self.usb_board_fallback = "ESP32-S3 DevKitC-1"
                        self.usb_chip_fallback = "ESP32-S3"
                    elif "esp32-c3" in desc:
                        self.usb_board_fallback = "ESP32-C3 SuperMini"
                        self.usb_chip_fallback = "ESP32-C3"
                    elif "esp32-c6" in desc:
                        self.usb_board_fallback = "ESP32-C6"
                        self.usb_chip_fallback = "ESP32-C6"
                    elif "esp32" in desc:
                        self.usb_board_fallback = "ESP32 DevKit V1"
                        self.usb_chip_fallback = "ESP32"
                    elif "stm32" in desc:
                        self.usb_board_fallback = "STM32 Board"
                        self.usb_chip_fallback = "STM32"
        except Exception as e:
            logger.warning(f"Error scanning USB fallback: {e}")

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
        import re
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

                # Parse board/SDK/firmware from lines
                lower_line = line.lower()
                
                # 1. Explicitly transmitted details
                board_match = re.search(r'\bboard(?:\s+name)?\s*:\s*([^\n\r]+)', lower_line)
                if board_match:
                    self.board_name = board_match.group(1).strip().title()

                chip_match = re.search(r'\bchip(?:\s+family)?\s*:\s*([^\n\r]+)', lower_line)
                if chip_match:
                    self.chip_family = chip_match.group(1).strip().upper()

                sdk_match = re.search(r'\bsdk(?:\s+version)?\s*:\s*([^\n\r]+)', lower_line)
                if sdk_match:
                    self.sdk_version = sdk_match.group(1).strip()

                fw_match = re.search(r'\bfirmware(?:\s+version)?\s*:\s*([^\n\r]+)', lower_line)
                if fw_match:
                    self.firmware_version = fw_match.group(1).strip()

                build_match = re.search(r'\bbuild(?:\s+version)?\s*:\s*([^\n\r]+)', lower_line)
                if build_match:
                    self.build_version = build_match.group(1).strip()

                app_match = re.search(r'\bapp(?:\s+version)?\s*:\s*([^\n\r]+)', lower_line)
                if app_match:
                    self.app_version = app_match.group(1).strip()

                uptime_match = re.search(r'\buptime\s*:\s*([^\n\r]+)', lower_line)
                if uptime_match:
                    self.uptime = uptime_match.group(1).strip()

                # 2. ESP-IDF boot logs details
                if "esp-rom:" in lower_line:
                    rom_match = re.search(r'esp-rom:([a-zA-Z0-9\-]+)', lower_line)
                    if rom_match and self.chip_family == "Unavailable":
                        rom_name = rom_match.group(1).strip().upper()
                        self.chip_family = rom_name.split("-")[0]
                        if "ESP32S3" in self.chip_family:
                            self.chip_family = "ESP32-S3"
                            if self.board_name == "Unknown Device":
                                self.board_name = "ESP32-S3 DevKitC-1"
                        elif "ESP32C3" in self.chip_family:
                            self.chip_family = "ESP32-C3"
                            if self.board_name == "Unknown Device":
                                self.board_name = "ESP32-C3 SuperMini"
                        elif "ESP32C6" in self.chip_family:
                            self.chip_family = "ESP32-C6"
                            if self.board_name == "Unknown Device":
                                self.board_name = "ESP32-C6"
                        elif "ESP32" in self.chip_family:
                            self.chip_family = "ESP32"
                            if self.board_name == "Unknown Device":
                                self.board_name = "ESP32 DevKit V1"

                if "esp-idf v" in lower_line:
                    match = re.search(r'esp-idf\s+(v[0-9a-zA-Z\.\-]+)', lower_line)
                    if match and self.sdk_version == "Unavailable":
                        self.sdk_version = match.group(1).strip().upper()

                if "compile time" in lower_line:
                    idx = lower_line.find("compile time")
                    if self.build_version == "Unavailable":
                        self.build_version = line[idx + len("compile time"):].strip()

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

    def get_connection_info(self):
        with self.lock:
            uptime_str = self.uptime
            if uptime_str == "Unavailable" and self.ser and self.ser.is_open and self.connection_start_time > 0.0:
                uptime_sec = int(time.time() - self.connection_start_time)
                uptime_str = f"{uptime_sec}s"
            
            board = "Unknown Device"
            if self.board_name != "Unknown Device":
                board = self.board_name
            elif getattr(self, "usb_board_fallback", "Unknown Device") != "Unknown Device":
                board = self.usb_board_fallback
                
            chip = "Unavailable"
            if self.chip_family != "Unavailable":
                chip = self.chip_family
            elif getattr(self, "usb_chip_fallback", "Unavailable") != "Unavailable":
                chip = self.usb_chip_fallback

            return {
                "connected": self.latest_telemetry["connected"],
                "board_name": board,
                "chip_family": chip,
                "port": self.port if self.port else "Unavailable",
                "baudrate": str(self.baudrate) if self.ser and self.ser.is_open else "Unavailable",
                "firmware_version": self.firmware_version,
                "build_version": self.build_version,
                "app_version": self.app_version,
                "sdk_version": self.sdk_version,
                "uptime": uptime_str
            }
