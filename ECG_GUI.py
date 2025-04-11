import sys
import numpy as np
import serial
import time
import threading
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.animation import FuncAnimation
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QLabel, QComboBox, QSpinBox,
                            QGridLayout, QFrame, QLCDNumber, QCheckBox, QGroupBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor
from scipy import signal


class ECGFilter:
    """Class for filtering ECG signals"""
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate
        
        # Design filters
        self.setup_filters()
        
        # Filter states for continuity between calls
        self.notch_zi = signal.lfilter_zi(self.notch_b, self.notch_a)
        self.lowpass_zi = signal.lfilter_zi(self.lowpass_b, self.lowpass_a)
        self.highpass_zi = signal.lfilter_zi(self.highpass_b, self.highpass_a)
        self.bandpass_zi = signal.lfilter_zi(self.bandpass_b, self.bandpass_a)
        
    def setup_filters(self):
        # 60 Hz Notch filter for power line interference
        notch_freq = 60.0  # 60 Hz
        quality_factor = 30.0  # Quality factor
        w0 = notch_freq / (self.sampling_rate/2)
        self.notch_b, self.notch_a = signal.iirnotch(w0, quality_factor)
        
        # Low-pass filter (for high frequency noise)
        cutoff_low = 50.0  # 50 Hz cutoff
        self.lowpass_b, self.lowpass_a = signal.butter(4, cutoff_low/(self.sampling_rate/2), 'lowpass')
        
        # High-pass filter (for baseline wander)
        cutoff_high = 0.5  # 0.5 Hz cutoff
        self.highpass_b, self.highpass_a = signal.butter(4, cutoff_high/(self.sampling_rate/2), 'highpass')
        
        # Bandpass filter (for ECG frequency range)
        lowcut = 0.5  # 0.5 Hz
        highcut = 40.0  # 40 Hz
        self.bandpass_b, self.bandpass_a = signal.butter(4, [lowcut/(self.sampling_rate/2), 
                                                         highcut/(self.sampling_rate/2)], 'bandpass')
    
    def apply_notch_filter(self, data):
        """Apply 60 Hz notch filter"""
        if isinstance(data, (int, float)):
            # Single value
            filtered, self.notch_zi = signal.lfilter(self.notch_b, self.notch_a, [float(data)], zi=self.notch_zi)
            return filtered[0]
        else:
            # Array of values
            filtered, self.notch_zi = signal.lfilter(self.notch_b, self.notch_a, np.array(data, dtype=float), zi=self.notch_zi*data[0])
            return filtered
    
    def apply_lowpass_filter(self, data):
        """Apply low-pass filter"""
        if isinstance(data, (int, float)):
            filtered, self.lowpass_zi = signal.lfilter(self.lowpass_b, self.lowpass_a, [float(data)], zi=self.lowpass_zi)
            return filtered[0]
        else:
            filtered, self.lowpass_zi = signal.lfilter(self.lowpass_b, self.lowpass_a, np.array(data, dtype=float), zi=self.lowpass_zi*data[0])
            return filtered
    
    def apply_highpass_filter(self, data):
        """Apply high-pass filter for baseline wander removal"""
        if isinstance(data, (int, float)):
            filtered, self.highpass_zi = signal.lfilter(self.highpass_b, self.highpass_a, [float(data)], zi=self.highpass_zi)
            return filtered[0]
        else:
            filtered, self.highpass_zi = signal.lfilter(self.highpass_b, self.highpass_a, np.array(data, dtype=float), zi=self.highpass_zi*data[0])
            return filtered
    
    def apply_bandpass_filter(self, data):
        """Apply bandpass filter for ECG frequency range"""
        if isinstance(data, (int, float)):
            filtered, self.bandpass_zi = signal.lfilter(self.bandpass_b, self.bandpass_a, [float(data)], zi=self.bandpass_zi)
            return filtered[0]
        else:
            filtered, self.bandpass_zi = signal.lfilter(self.bandpass_b, self.bandpass_a, np.array(data, dtype=float), zi=self.bandpass_zi*data[0])
            return filtered
    

class SerialReader(QObject):
    data_received = pyqtSignal(int, int)  # Raw and filtered values
    beat_detected = pyqtSignal()
    
    def __init__(self, port='COM3', baudrate=9600):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_running = False
        
        # Data storage
        self.raw_data = deque(maxlen=2000)
        self.filtered_data = deque(maxlen=2000)
        self.timestamps = deque(maxlen=2000)
        
        # Beat detection parameters
        self.threshold = 550  # Initial threshold
        self.adaptive_threshold = 550  # Adaptive threshold for beat detection
        self.last_beat_time = 0
        self.beat_count = 0
        self.min_beat_interval = 0.3  # Minimum time between beats (in seconds)
        self.refractory_period = 0.2  # Time after a beat where we ignore signals
        
        # Filtering options
        self.use_notch_filter = True
        self.use_lowpass_filter = True
        self.use_highpass_filter = True
        self.use_bandpass_filter = False  # Off by default as it's more aggressive
        
        # Create filters
        self.ecg_filter = ECGFilter()
        
        # Adaptive threshold parameters
        self.window_size = 500  # Window size for adaptive threshold calculation
        self.threshold_factor = 0.6  # Threshold = factor * peak value
        
    def connect(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate)
            return True
        except serial.SerialException as e:
            print(f"Error connecting to serial port: {e}")
            return False
    
    def disconnect(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
    
    def start_reading(self):
        if not self.serial_conn or not self.serial_conn.is_open:
            if not self.connect():
                return False
        
        self.is_running = True
        self.thread = threading.Thread(target=self._read_loop)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def stop_reading(self):
        self.is_running = False
        if hasattr(self, 'thread') and self.thread:
            self.thread.join(timeout=1.0)
        self.disconnect()
    
    def set_filter_options(self, notch=True, lowpass=True, highpass=True, bandpass=False):
        self.use_notch_filter = notch
        self.use_lowpass_filter = lowpass
        self.use_highpass_filter = highpass
        self.use_bandpass_filter = bandpass
    
    def apply_filters(self, value):
        """Apply selected filters to the raw ECG value"""
        filtered_value = float(value)
        
        if self.use_notch_filter:
            filtered_value = self.ecg_filter.apply_notch_filter(filtered_value)
            
        if self.use_highpass_filter:
            filtered_value = self.ecg_filter.apply_highpass_filter(filtered_value)
            
        if self.use_lowpass_filter:
            filtered_value = self.ecg_filter.apply_lowpass_filter(filtered_value)
            
        if self.use_bandpass_filter:
            filtered_value = self.ecg_filter.apply_bandpass_filter(filtered_value)
            
        return int(filtered_value)
    
    def update_adaptive_threshold(self):
        """Update the adaptive threshold based on recent signal"""
        if len(self.filtered_data) >= 50:
            # Get last window_size samples, or as many as available
            window = list(self.filtered_data)[-min(self.window_size, len(self.filtered_data)):]
            
            if len(window) > 0:
                signal_range = max(window) - min(window)
                signal_mean = np.mean(window)
                
                # Update threshold: mean + portion of the signal range
                self.adaptive_threshold = signal_mean + (signal_range * self.threshold_factor)
    
    def _read_loop(self):
        buffer = ""
        while self.is_running:
            try:
                if self.serial_conn.in_waiting > 0:
                    # Read data and append to buffer
                    char = self.serial_conn.read(1).decode('utf-8')
                    if char == '\n':
                        # Process complete line
                        line = buffer.strip()
                        buffer = ""
                        
                        # Check if line is a valid number and not a leads-off indicator ('!')
                        if line != '!' and line.strip():
                            try:
                                raw_value = int(line)
                                current_time = time.time()
                                
                                # Store the raw value
                                self.raw_data.append(raw_value)
                                self.timestamps.append(current_time)
                                
                                # Apply filters
                                filtered_value = self.apply_filters(raw_value)
                                self.filtered_data.append(filtered_value)
                                
                                # Update adaptive threshold periodically
                                if len(self.raw_data) % 20 == 0:
                                    self.update_adaptive_threshold()
                                
                                # Detect beats using filtered data
                                self._detect_beat(filtered_value, current_time)
                                
                                # Emit signal with the raw and filtered values
                                self.data_received.emit(raw_value, filtered_value)
                            except ValueError:
                                pass  # Ignore non-numeric data
                    else:
                        buffer += char
            except Exception as e:
                print(f"Error reading serial data: {e}")
                self.is_running = False
                break
            time.sleep(0.001)  # Small delay to prevent high CPU usage
    
    def _detect_beat(self, value, current_time):
        """Detect heartbeats using adaptive thresholding"""
        # Skip if we're in the refractory period after a beat
        if current_time - self.last_beat_time < self.refractory_period:
            return
            
        # Check if the value crosses threshold and enough time has passed since last beat
        if value > self.adaptive_threshold and (current_time - self.last_beat_time) > self.min_beat_interval:
            self.beat_count += 1
            self.last_beat_time = current_time
            self.beat_detected.emit()
    
    def get_heart_rate(self):
        """Calculate heart rate based on recent beats"""
        if len(self.timestamps) < 2:
            return 0
            
        # Calculate average heart rate from R-R intervals
        rr_intervals = []
        beat_times = []
        
        # Find times of recent beats (last 10 seconds)
        recent_time = self.timestamps[-1] - 10 if self.timestamps else time.time() - 10
        
        # Extract beat timestamps
        for i in range(1, len(self.timestamps)):
            if self.timestamps[i] > recent_time:
                if getattr(self, f"_is_beat_at_{i}", False):
                    beat_times.append(self.timestamps[i])
        
        # Calculate R-R intervals
        for i in range(1, len(beat_times)):
            rr_interval = beat_times[i] - beat_times[i-1]
            if 0.3 <= rr_interval <= 2.0:  # Only accept physiologically plausible intervals
                rr_intervals.append(rr_interval)
        
        # Calculate heart rate
        if len(rr_intervals) > 0:
            avg_rr = np.mean(rr_intervals)
            return int(60 / avg_rr) if avg_rr > 0 else 0
        
        # Fallback: calculate from beat count in the last 10 seconds
        recent_beat_times = [t for t in self.timestamps if t > recent_time]
        if len(recent_beat_times) >= 2:
            time_span = recent_beat_times[-1] - recent_beat_times[0]
            beat_count = len(recent_beat_times) - 1
            return int((beat_count / time_span) * 60) if time_span > 0 else 0
        
        return 0


class ECGPlot(FigureCanvas):
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        
        self.setParent(parent)
        
        # Setup data storage
        self.raw_data = deque(maxlen=1000)
        self.filtered_data = deque(maxlen=1000)
        self.times = deque(maxlen=1000)
        self.beats = []  # Store beat times for marking
        self.start_time = time.time()
        
        # Initialize line objects
        self.raw_line, = self.ax.plot([], [], lw=1.0, color='lightgray', label='Raw')
        self.filtered_line, = self.ax.plot([], [], lw=1.5, color='g', label='Filtered')
        
        # Setup axes
        self.ax.set_ylim(300, 700)  # Adjust based on your ECG signal range
        self.ax.set_xlim(0, 5)  # Display 5 seconds of data
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('ECG Value')
        self.ax.set_title('Real-time ECG')
        self.ax.grid(True)
        self.ax.legend(loc='upper right')
        
        self.fig.tight_layout()
        
        # Setup animation
        self.animation = FuncAnimation(
            self.fig, self.update_plot, interval=50, blit=True)
        
        # Display flags
        self.show_raw = True
        self.show_filtered = True
    
    def add_data(self, raw_value, filtered_value):
        current_time = time.time() - self.start_time
        self.raw_data.append(raw_value)
        self.filtered_data.append(filtered_value)
        self.times.append(current_time)
    
    def update_plot(self, frame):
        if len(self.times) > 0:
            current_time = self.times[-1]
            self.ax.set_xlim(max(0, current_time - 5), current_time + 0.1)
            
            # Update data
            if self.show_raw:
                self.raw_line.set_data(self.times, self.raw_data)
            else:
                self.raw_line.set_data([], [])
                
            if self.show_filtered:
                self.filtered_line.set_data(self.times, self.filtered_data)
            else:
                self.filtered_line.set_data([], [])
            
            # Clean old beat markers (keep only recent ones)
            for line in self.ax.lines[2:]:  # Skip our two main lines
                if line.get_xdata()[0] < current_time - 5:
                    line.remove()
        
        return tuple(self.ax.lines)  # Return all lines
    
    def mark_beat(self):
        if len(self.times) > 0:
            # Mark the beat with a vertical line
            beat_time = self.times[-1]
            self.beats.append(beat_time)
            self.ax.axvline(x=beat_time, color='r', alpha=0.5, linewidth=1)
    
    def set_display_options(self, show_raw=True, show_filtered=True):
        self.show_raw = show_raw
        self.show_filtered = show_filtered


class ECGMonitorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Monitor with Filters")
        self.setGeometry(100, 100, 1200, 700)
        
        # Set up the serial reader
        self.serial_reader = SerialReader()
        self.serial_reader.data_received.connect(self.update_data)
        self.serial_reader.beat_detected.connect(self.on_beat_detected)
        
        # Heartbeat parameters
        self.beats_per_minute = 0
        self.beat_count = 0
        self.last_beat_time = time.time()
        
        # Alarm thresholds
        self.low_hr_threshold = 60   # Bradycardia threshold
        self.high_hr_threshold = 100  # Tachycardia threshold
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Top section: Connection controls
        connection_frame = QFrame()
        connection_frame.setFrameShape(QFrame.StyledPanel)
        connection_layout = QHBoxLayout()
        connection_frame.setLayout(connection_layout)
        
        # Serial port selection
        self.port_label = QLabel("Port:")
        self.port_combo = QComboBox()
        # Add common port names
        self.port_combo.addItems(["COM3", "COM4", "COM5", "/dev/ttyUSB0", "/dev/ttyACM0"])
        self.port_combo.setEditable(True)
        
        # Connect/Disconnect button
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.toggle_connection)
        
        connection_layout.addWidget(self.port_label)
        connection_layout.addWidget(self.port_combo)
        connection_layout.addWidget(self.connect_button)
        connection_layout.addStretch()
        
        # Filter controls
        filter_box = QGroupBox("Filter Settings")
        filter_layout = QGridLayout()
        filter_box.setLayout(filter_layout)
        
        # Filter checkboxes
        self.notch_filter_cb = QCheckBox("60 Hz Notch Filter")
        self.notch_filter_cb.setChecked(True)
        self.notch_filter_cb.stateChanged.connect(self.update_filter_settings)
        
        self.lowpass_filter_cb = QCheckBox("Low-pass Filter (50 Hz)")
        self.lowpass_filter_cb.setChecked(True)
        self.lowpass_filter_cb.stateChanged.connect(self.update_filter_settings)
        
        self.highpass_filter_cb = QCheckBox("High-pass Filter (0.5 Hz)")
        self.highpass_filter_cb.setChecked(True)
        self.highpass_filter_cb.stateChanged.connect(self.update_filter_settings)
        
        self.bandpass_filter_cb = QCheckBox("Bandpass Filter (0.5-40 Hz)")
        self.bandpass_filter_cb.setChecked(False)
        self.bandpass_filter_cb.stateChanged.connect(self.update_filter_settings)
        
        # Display options
        self.show_raw_cb = QCheckBox("Show Raw Signal")
        self.show_raw_cb.setChecked(True)
        self.show_raw_cb.stateChanged.connect(self.update_display_settings)
        
        self.show_filtered_cb = QCheckBox("Show Filtered Signal")
        self.show_filtered_cb.setChecked(True)
        self.show_filtered_cb.stateChanged.connect(self.update_display_settings)
        
        filter_layout.addWidget(self.notch_filter_cb, 0, 0)
        filter_layout.addWidget(self.lowpass_filter_cb, 0, 1)
        filter_layout.addWidget(self.highpass_filter_cb, 1, 0)
        filter_layout.addWidget(self.bandpass_filter_cb, 1, 1)
        filter_layout.addWidget(self.show_raw_cb, 0, 2)
        filter_layout.addWidget(self.show_filtered_cb, 1, 2)
        
        connection_layout.addWidget(filter_box)
        
        # Middle section: ECG plot
        self.ecg_plot = ECGPlot(width=10, height=4)
        
        # Bottom section: Heart rate and alarm thresholds
        metrics_frame = QFrame()
        metrics_frame.setFrameShape(QFrame.StyledPanel)
        metrics_layout = QGridLayout()
        metrics_frame.setLayout(metrics_layout)
        
        # Heart rate display
        hr_label = QLabel("Heart Rate:")
        hr_label.setFont(QFont("Arial", 14))
        self.hr_display = QLCDNumber()
        self.hr_display.setDigitCount(3)
        self.hr_display.display(0)
        self.hr_display.setSegmentStyle(QLCDNumber.Flat)
        
        # Alarm settings
        low_hr_label = QLabel("Low HR Alarm:")
        self.low_hr_spin = QSpinBox()
        self.low_hr_spin.setRange(30, 100)
        self.low_hr_spin.setValue(self.low_hr_threshold)
        self.low_hr_spin.valueChanged.connect(self.update_low_threshold)
        
        high_hr_label = QLabel("High HR Alarm:")
        self.high_hr_spin = QSpinBox()
        self.high_hr_spin.setRange(60, 200)
        self.high_hr_spin.setValue(self.high_hr_threshold)
        self.high_hr_spin.valueChanged.connect(self.update_high_threshold)
        
        # Status display
        status_label = QLabel("Status:")
        self.status_display = QLabel("Not Connected")
        self.status_display.setFrameShape(QFrame.Box)
        self.status_display.setAlignment(Qt.AlignCenter)
        self.status_display.setMinimumWidth(150)
        
        # Add widgets to metrics layout
        metrics_layout.addWidget(hr_label, 0, 0)
        metrics_layout.addWidget(self.hr_display, 0, 1)
        metrics_layout.addWidget(QLabel("BPM"), 0, 2)
        metrics_layout.addWidget(low_hr_label, 0, 3)
        metrics_layout.addWidget(self.low_hr_spin, 0, 4)
        metrics_layout.addWidget(high_hr_label, 0, 5)
        metrics_layout.addWidget(self.high_hr_spin, 0, 6)
        metrics_layout.addWidget(status_label, 0, 7)
        metrics_layout.addWidget(self.status_display, 0, 8)
        
        # Add all sections to main layout
        main_layout.addWidget(connection_frame)
        main_layout.addWidget(self.ecg_plot)
        main_layout.addWidget(metrics_frame)
        
        # Set up timer for heart rate calculation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_heart_rate)
        self.timer.start(1000)  # Update every second
    
    def toggle_connection(self):
        if self.connect_button.text() == "Connect":
            # Set port from combo box
            self.serial_reader.port = self.port_combo.currentText()
            
            # Try to connect
            if self.serial_reader.start_reading():
                self.connect_button.setText("Disconnect")
                self.status_display.setText("Connected")
                self.status_display.setStyleSheet("background-color: #c8e6c9;")  # Light green
            else:
                self.status_display.setText("Connection Failed")
                self.status_display.setStyleSheet("background-color: #ffcdd2;")  # Light red
        else:
            # Disconnect
            self.serial_reader.stop_reading()
            self.connect_button.setText("Connect")
            self.status_display.setText("Disconnected")
            self.status_display.setStyleSheet("")
    
    def update_data(self, raw_value, filtered_value):
        # Add new data point to the plot
        self.ecg_plot.add_data(raw_value, filtered_value)
    
    def on_beat_detected(self):
        # Mark beat on the plot
        self.ecg_plot.mark_beat()
        
        # Update beat count
        self.beat_count += 1
        current_time = time.time()
        
        # Calculate instantaneous heart rate from the interval
        interval = current_time - self.last_beat_time
        if interval > 0:
            instant_hr = int(60 / interval)
            # Only update if the rate is physiologically plausible (30-220 BPM)
            if 30 <= instant_hr <= 220:
                self.beats_per_minute = instant_hr
        
        self.last_beat_time = current_time
    
    def update_heart_rate(self):
        # Get the heart rate from serial reader
        hr = self.serial_reader.get_heart_rate()
        if hr > 0:
            self.beats_per_minute = hr
        
        # Update display
        self.hr_display.display(self.beats_per_minute)
        
        # Check for alarms
        if self.beats_per_minute < self.low_hr_threshold and self.beats_per_minute > 0:
            self.status_display.setText("BRADYCARDIA ALARM!")
            self.status_display.setStyleSheet("background-color: #ffcdd2; color: red; font-weight: bold;")
        elif self.beats_per_minute > self.high_hr_threshold:
            self.status_display.setText("TACHYCARDIA ALARM!")
            self.status_display.setStyleSheet("background-color: #ffcdd2; color: red; font-weight: bold;")
        elif self.connect_button.text() == "Disconnect":
            self.status_display.setText("Normal")
            self.status_display.setStyleSheet("background-color: #c8e6c9;")  # Light green
    
    def update_low_threshold(self, value):
        self.low_hr_threshold = value
    
    def update_high_threshold(self, value):
        self.high_hr_threshold = value
    
    def update_filter_settings(self):
        # Update filter settings in the serial reader
        self.serial_reader.set_filter_options(
            notch=self.notch_filter_cb.isChecked(),
            lowpass=self.lowpass_filter_cb.isChecked(),
            highpass=self.highpass_filter_cb.isChecked(),
            bandpass=self.bandpass_filter_cb.isChecked()
        )
    
    def update_display_settings(self):
        # Update display settings in the plot
        self.ecg_plot.set_display_options(
            show_raw=self.show_raw_cb.isChecked(),
            show_filtered=self.show_filtered_cb.isChecked()
        )
    
    def closeEvent(self, event):
        # Clean up when window is closed
        self.serial_reader.stop_reading()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ECGMonitorApp()
    window.show()
    sys.exit(app.exec_())