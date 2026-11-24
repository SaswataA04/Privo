
import json
import os


class SettingsManager:

    # Default application configuration.
    DEFAULT_SETTINGS = {
        "confidence_threshold": 65,
        "trigger_delay": 1.0,
        "clear_delay": 1.5,
        "camera_index": 0
    }

    def __init__(self):

        # Store settings beside this Python file.
        self.base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.settings_path = os.path.join(
            self.base_dir,
            "settings.json"
        )

        self.settings = self.load_settings()

    def load_settings(self):

        """
        Load settings from JSON.

        If the file does not exist or contains invalid data,
        use the default configuration.
        """

        if not os.path.exists(self.settings_path):
            self.save_settings(self.DEFAULT_SETTINGS)
            return self.DEFAULT_SETTINGS.copy()

        try:
            with open(
                self.settings_path,
                "r",
                encoding="utf-8"
            ) as file:

                loaded = json.load(file)

            # Merge loaded settings with defaults.
            # This supports newly added settings in future versions.
            settings = self.DEFAULT_SETTINGS.copy()
            settings.update(loaded)

            return self.validate_settings(settings)

        except (json.JSONDecodeError, OSError, TypeError, ValueError):

            return self.DEFAULT_SETTINGS.copy()

    def validate_settings(self, settings):

        """Validate and normalize configuration values."""

        defaults = self.DEFAULT_SETTINGS

        try:
            threshold = int(settings["confidence_threshold"])
            trigger = float(settings["trigger_delay"])
            clear = float(settings["clear_delay"])
            camera = int(settings["camera_index"])

            if not 10 <= threshold <= 150:
                raise ValueError("Invalid recognition threshold")

            if not 0.2 <= trigger <= 10:
                raise ValueError("Invalid warning delay")

            if not 0.2 <= clear <= 10:
                raise ValueError("Invalid clear delay")

            if not 0 <= camera <= 10:
                raise ValueError("Invalid camera index")

            return {
                "confidence_threshold": threshold,
                "trigger_delay": trigger,
                "clear_delay": clear,
                "camera_index": camera
            }

        except (KeyError, TypeError, ValueError):
            return defaults.copy()

    def save_settings(self, settings):

        """Validate and write settings to the JSON file."""

        validated = self.validate_settings(settings)

        with open(
            self.settings_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                validated,
                file,
                indent=4
            )

        self.settings = validated

        return validated

    def get_settings(self):

        """Return a copy so callers cannot modify internal state."""

        return self.settings.copy()
