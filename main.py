
import sys

from startup_manager import StartupManager
from camera_worker import CameraWorker
from settings_manager import SettingsManager

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QSystemTrayIcon,
    QMenu,
    QMessageBox,
    QDialog,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QDialogButtonBox,
    QGroupBox
)


# =============================================================
# SETTINGS DIALOG
# =============================================================

class SettingsDialog(QDialog):

    def __init__(self, current_settings, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Privacy Guard Settings")
        self.setMinimumWidth(420)

        self.setStyleSheet("""
            QDialog {
                background-color: #0b1120;
                color: #e2e8f0;
            }

            QLabel {
                color: #cbd5e1;
            }

            QSpinBox, QDoubleSpinBox {
                background-color: #1e293b;
                color: white;
                border: 1px solid #475569;
                border-radius: 5px;
                padding: 7px;
                min-width: 100px;
            }

            QGroupBox {
                color: #93c5fd;
                border: 1px solid #334155;
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px;
                font-weight: bold;
            }

            QPushButton {
                background-color: #2563eb;
                color: white;
                padding: 8px 16px;
                border-radius: 5px;
            }

            QPushButton:hover {
                background-color: #1d4ed8;
            }
        """)

        main_layout = QVBoxLayout(self)

        # -----------------------------------------------------
        # FACE RECOGNITION SETTINGS
        # -----------------------------------------------------

        recognition_group = QGroupBox("Face Recognition")
        recognition_layout = QFormLayout(recognition_group)

        self.threshold_input = QSpinBox()
        self.threshold_input.setRange(10, 150)
        self.threshold_input.setValue(
            current_settings["confidence_threshold"]
        )

        self.threshold_input.setToolTip(
            "Higher values make recognition more permissive. "
            "This can increase false acceptance."
        )

        recognition_layout.addRow(
            "LBPH threshold:",
            self.threshold_input
        )

        threshold_note = QLabel(
            "Lower = stricter matching. Higher = more permissive."
        )
        threshold_note.setWordWrap(True)
        threshold_note.setStyleSheet(
            "color: #94a3b8; font-size: 11px;"
        )

        recognition_layout.addRow("", threshold_note)
        main_layout.addWidget(recognition_group)

        # -----------------------------------------------------
        # VISUAL ALERT SETTINGS
        # -----------------------------------------------------

        alert_group = QGroupBox("Visual Alert Timing")
        alert_layout = QFormLayout(alert_group)

        self.trigger_input = QDoubleSpinBox()
        self.trigger_input.setRange(0.2, 10.0)
        self.trigger_input.setSingleStep(0.1)
        self.trigger_input.setDecimals(1)
        self.trigger_input.setSuffix(" sec")
        self.trigger_input.setValue(
            current_settings["trigger_delay"]
        )

        self.clear_input = QDoubleSpinBox()
        self.clear_input.setRange(0.2, 10.0)
        self.clear_input.setSingleStep(0.1)
        self.clear_input.setDecimals(1)
        self.clear_input.setSuffix(" sec")
        self.clear_input.setValue(
            current_settings["clear_delay"]
        )

        alert_layout.addRow("Warning delay:", self.trigger_input)
        alert_layout.addRow("Clear delay:", self.clear_input)

        main_layout.addWidget(alert_group)

        # -----------------------------------------------------
        # CAMERA SETTINGS
        # -----------------------------------------------------

        camera_group = QGroupBox("Camera")
        camera_layout = QFormLayout(camera_group)

        self.camera_input = QSpinBox()
        self.camera_input.setRange(0, 10)
        self.camera_input.setValue(
            current_settings["camera_index"]
        )

        self.camera_input.setToolTip(
            "0 is usually the built-in webcam. "
            "Other indexes may refer to external cameras."
        )

        camera_layout.addRow("Camera index:", self.camera_input)

        camera_note = QLabel(
            "Camera index 0 is usually the default webcam."
        )
        camera_note.setStyleSheet(
            "color: #94a3b8; font-size: 11px;"
        )

        camera_layout.addRow("", camera_note)
        main_layout.addWidget(camera_group)

        # -----------------------------------------------------
        # SAVE / CANCEL BUTTONS
        # -----------------------------------------------------

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        main_layout.addWidget(button_box)

    def get_settings(self):
        """Return the settings selected by the user."""

        return {
            "confidence_threshold": self.threshold_input.value(),
            "trigger_delay": self.trigger_input.value(),
            "clear_delay": self.clear_input.value(),
            "camera_index": self.camera_input.value()
        }


# =============================================================
# MAIN APPLICATION WINDOW
# =============================================================

class PrivacyGuardWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        # Startup manager handles Windows login startup.
        self.startup_manager = StartupManager()

        # Camera worker reference.
        self.worker = None

        # Prevents closeEvent from hiding the app during quit.
        self.closing_application = False

        # Load saved settings.
        self.settings_manager = SettingsManager()
        self.settings = self.settings_manager.get_settings()

        self.setWindowTitle("Privacy Guard")
        self.setMinimumSize(900, 700)

        # Build the dashboard and system tray.
        self.build_interface()
        self.setup_tray()

        # Create custom visual-only warning popup.
        self.alert_popup = None
        self.create_alert_popup()

    # ---------------------------------------------------------
    # BUILD DASHBOARD
    # ---------------------------------------------------------

    def build_interface(self):

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(15)

        # Application heading.
        heading = QLabel("PRIVACY GUARD")
        heading.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #e8edf5;
        """)

        subtitle = QLabel(
            "Local webcam-based privacy monitoring"
        )
        subtitle.setStyleSheet("color: #94a3b8;")

        layout.addWidget(heading)
        layout.addWidget(subtitle)

        # -----------------------------------------------------
        # PROTECTION STATUS
        # -----------------------------------------------------

        self.status_label = QLabel("MONITORING STOPPED")
        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.status_label.setMinimumHeight(65)

        self.set_status_style("stopped")
        layout.addWidget(self.status_label)

        # -----------------------------------------------------
        # CAMERA PREVIEW
        # -----------------------------------------------------

        self.camera_label = QLabel(
            "Camera preview will appear here"
        )

        self.camera_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.camera_label.setMinimumHeight(400)

        self.camera_label.setStyleSheet("""
            background-color: #111827;
            color: #94a3b8;
            border: 1px solid #334155;
            border-radius: 12px;
            font-size: 16px;
        """)

        layout.addWidget(self.camera_label)

        # -----------------------------------------------------
        # FACE COUNTERS
        # -----------------------------------------------------

        self.counter_label = QLabel(
            "Authorized: 0       Unknown: 0"
        )

        self.counter_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.counter_label.setStyleSheet("""
            color: #cbd5e1;
            font-size: 15px;
        """)

        layout.addWidget(self.counter_label)

        # -----------------------------------------------------
        # DASHBOARD BUTTONS
        # -----------------------------------------------------

        button_layout = QHBoxLayout()

        self.start_button = QPushButton("Start Monitoring")
        self.pause_button = QPushButton("Pause Monitoring")
        self.settings_button = QPushButton("Settings")
        self.startup_button = QPushButton()

        self.start_button.clicked.connect(
            self.start_monitoring
        )

        self.pause_button.clicked.connect(
            self.toggle_pause
        )

        self.settings_button.clicked.connect(
            self.open_settings
        )

        self.startup_button.clicked.connect(
            self.toggle_startup
        )

        self.pause_button.setEnabled(False)

        # Update startup button according to current setting.
        self.update_startup_button()

        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #15803d;
                color: white;
                padding: 13px;
                border-radius: 7px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #166534;
            }

            QPushButton:disabled {
                background-color: #334155;
                color: #64748b;
            }
        """)

        self.pause_button.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: white;
                padding: 13px;
                border-radius: 7px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #475569;
            }
        """)

        self.settings_button.setStyleSheet("""
            QPushButton {
                background-color: #1d4ed8;
                color: white;
                padding: 13px;
                border-radius: 7px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #1e40af;
            }

            QPushButton:disabled {
                background-color: #334155;
                color: #64748b;
            }
        """)

        self.startup_button.setStyleSheet("""
            QPushButton {
                background-color: #475569;
                color: white;
                padding: 13px;
                border-radius: 7px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #64748b;
            }
        """)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.settings_button)
        button_layout.addWidget(self.startup_button)

        layout.addLayout(button_layout)

        # -----------------------------------------------------
        # FOOTER
        # -----------------------------------------------------

        footer = QLabel(
            "Processing stays on this device. "
            "No webcam footage is recorded."
        )

        footer.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        footer.setStyleSheet(
            "color: #64748b; font-size: 11px;"
        )

        layout.addWidget(footer)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0b1120;
            }
        """)

    # ---------------------------------------------------------
    # STATUS STYLING
    # ---------------------------------------------------------

    def set_status_style(self, state):

        if state == "protected":
            self.status_label.setText("PROTECTED")
            background = "#14532d"
            foreground = "#bbf7d0"

        elif state == "warning":
            self.status_label.setText(
                "PRIVACY WARNING — UNKNOWN FACE DETECTED"
            )
            background = "#7f1d1d"
            foreground = "#fecaca"

        elif state == "paused":
            self.status_label.setText("MONITORING PAUSED")
            background = "#78350f"
            foreground = "#fde68a"

        else:
            self.status_label.setText("MONITORING STOPPED")
            background = "#334155"
            foreground = "#e2e8f0"

        self.status_label.setStyleSheet(f"""
            background-color: {background};
            color: {foreground};
            border-radius: 10px;
            font-size: 19px;
            font-weight: bold;
        """)

    # ---------------------------------------------------------
    # OPEN SETTINGS
    # ---------------------------------------------------------

    def open_settings(self):

        # Do not allow settings changes while camera is active.
        if self.worker and self.worker.isRunning():

            QMessageBox.information(
                self,
                "Monitoring Active",
                "Please stop monitoring before changing settings."
            )
            return

        dialog = SettingsDialog(self.settings, self)

        if dialog.exec() == QDialog.DialogCode.Accepted:

            new_settings = dialog.get_settings()

            try:
                self.settings = self.settings_manager.save_settings(
                    new_settings
                )

                QMessageBox.information(
                    self,
                    "Settings Saved",
                    "Your settings have been saved successfully."
                )

            except (OSError, ValueError) as error:

                QMessageBox.critical(
                    self,
                    "Settings Error",
                    f"Unable to save settings:\n{error}"
                )

    # ---------------------------------------------------------
    # START MONITORING
    # ---------------------------------------------------------

    def start_monitoring(self):

        if self.worker and self.worker.isRunning():
            return

        try:
            self.worker = CameraWorker(
                settings=self.settings
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Initialization Error",
                str(error)
            )
            return

        # Connect worker signals to dashboard methods.
        self.worker.frame_ready.connect(
            self.update_camera_preview
        )

        self.worker.status_changed.connect(
            self.update_status
        )

        self.worker.error_occurred.connect(
            self.show_error
        )

        self.worker.finished.connect(
            self.monitoring_finished
        )

        # Update button states.
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.settings_button.setEnabled(False)

        self.pause_button.setText("Pause Monitoring")

        self.hide_warning_notification()
        self.set_status_style("protected")

        self.worker.start()

    # ---------------------------------------------------------
    # PAUSE / RESUME MONITORING
    # ---------------------------------------------------------

    def toggle_pause(self):

        if not self.worker or not self.worker.isRunning():
            return

        if not self.worker.paused:

            self.worker.pause_monitoring()

            self.pause_button.setText("Resume Monitoring")

            self.hide_warning_notification()
            self.set_status_style("paused")

        else:

            self.worker.resume_monitoring()

            self.pause_button.setText("Pause Monitoring")
            self.set_status_style("protected")

    # ---------------------------------------------------------
    # UPDATE CAMERA PREVIEW
    # ---------------------------------------------------------

    def update_camera_preview(self, image):

        pixmap = QPixmap.fromImage(image)

        self.camera_label.setPixmap(
            pixmap.scaled(
                self.camera_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    # ---------------------------------------------------------
    # CUSTOM VISUAL-ONLY ALERT POPUP
    # ---------------------------------------------------------

    def create_alert_popup(self):

        self.alert_popup = QLabel(
            "PRIVACY WARNING\nUNKNOWN FACE DETECTED"
        )

        self.alert_popup.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )

        self.alert_popup.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.alert_popup.setFixedSize(340, 100)

        self.alert_popup.setStyleSheet("""
            QLabel {
                background-color: #991b1b;
                color: white;
                border: 2px solid #ef4444;
                border-radius: 12px;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
            }
        """)

        self.alert_popup.hide()

    def show_warning_notification(self):

        if not self.alert_popup:
            return

        screen = QApplication.primaryScreen()

        if screen is None:
            return

        available = screen.availableGeometry()

        x = (
            available.right()
            - self.alert_popup.width()
            - 25
        )

        y = (
            available.bottom()
            - self.alert_popup.height()
            - 25
        )

        self.alert_popup.move(x, y)
        self.alert_popup.show()
        self.alert_popup.raise_()

    def hide_warning_notification(self):

        if self.alert_popup:
            self.alert_popup.hide()

    # ---------------------------------------------------------
    # UPDATE STATUS AND FACE COUNTERS
    # ---------------------------------------------------------

    def update_status(
        self,
        status,
        authorized_count,
        unknown_count
    ):

        self.counter_label.setText(
            f"Authorized: {authorized_count}       "
            f"Unknown: {unknown_count}"
        )

        # Do not overwrite the paused status.
        if self.worker and self.worker.paused:
            return

        if status == "PRIVACY WARNING":

            self.set_status_style("warning")
            self.show_warning_notification()

        else:

            self.set_status_style("protected")
            self.hide_warning_notification()

    # ---------------------------------------------------------
    # ERROR HANDLING
    # ---------------------------------------------------------

    def show_error(self, message):

        self.hide_warning_notification()
        self.set_status_style("stopped")

        QMessageBox.critical(
            self,
            "Privacy Guard Error",
            message
        )

    # ---------------------------------------------------------
    # WORKER FINISHED
    # ---------------------------------------------------------

    def monitoring_finished(self):

        self.hide_warning_notification()

        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.settings_button.setEnabled(True)

        self.pause_button.setText("Pause Monitoring")

        self.set_status_style("stopped")

    # ---------------------------------------------------------
    # WINDOWS STARTUP BUTTON
    # ---------------------------------------------------------

    def update_startup_button(self):

        if self.startup_manager.is_enabled():

            self.startup_button.setText(
                "Disable Windows Startup"
            )

        else:

            self.startup_button.setText(
                "Enable Windows Startup"
            )

    def toggle_startup(self):

        try:

            if self.startup_manager.is_enabled():

                self.startup_manager.disable()

                QMessageBox.information(
                    self,
                    "Windows Startup",
                    "Privacy Guard will no longer launch "
                    "automatically when you sign in."
                )

            else:

                self.startup_manager.enable()

                QMessageBox.information(
                    self,
                    "Windows Startup",
                    "Privacy Guard will launch automatically "
                    "when you sign in to Windows."
                )

            self.update_startup_button()

        except (OSError, KeyError) as error:

            QMessageBox.critical(
                self,
                "Startup Error",
                str(error)
            )

    # ---------------------------------------------------------
    # SYSTEM TRAY
    # ---------------------------------------------------------

    def setup_tray(self):

        self.tray = QSystemTrayIcon(self)
        self.tray.setToolTip("Privacy Guard")

        menu = QMenu()

        open_action = QAction(
            "Open Privacy Guard",
            self
        )

        open_action.triggered.connect(
            self.show_window
        )

        quit_action = QAction(
            "Quit Privacy Guard",
            self
        )

        quit_action.triggered.connect(
            self.quit_application
        )

        menu.addAction(open_action)
        menu.addSeparator()
        menu.addAction(quit_action)

        self.tray.setContextMenu(menu)

        self.tray.activated.connect(
            self.tray_activated
        )

        self.tray.show()

    def tray_activated(self, reason):

        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.show_window()

    def show_window(self):

        self.show()
        self.raise_()
        self.activateWindow()

    # ---------------------------------------------------------
    # CLOSE WINDOW
    # ---------------------------------------------------------

    def closeEvent(self, event):

        if self.closing_application:
            event.accept()
            return

        # Closing the dashboard hides it.
        # Monitoring continues in the background.
        event.ignore()
        self.hide()

    # ---------------------------------------------------------
    # QUIT APPLICATION
    # ---------------------------------------------------------

    def quit_application(self):

        self.closing_application = True

        self.hide_warning_notification()

        if self.worker and self.worker.isRunning():
            self.worker.stop_monitoring()

        self.tray.hide()

        QApplication.quit()


# =============================================================
# APPLICATION ENTRY POINT
# =============================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    app.setApplicationName("Privacy Guard")

    # Keep the process alive when the dashboard is hidden.
    app.setQuitOnLastWindowClosed(False)

    window = PrivacyGuardWindow()

    # Windows startup launcher passes --minimized.
    if "--minimized" in sys.argv:
        window.hide()
    else:
        window.show()

    sys.exit(app.exec())
