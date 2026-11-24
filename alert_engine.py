
import time


class AlertEngine:
    """
    Controls when a privacy warning starts and clears.

    A warning starts only when an unknown face remains
    visible for trigger_delay seconds.

    It clears only after no unknown face is detected
    for clear_delay seconds.
    """

    def __init__(self, trigger_delay=1.0, clear_delay=1.5):
        self.trigger_delay = trigger_delay
        self.clear_delay = clear_delay

        self.unknown_since = None
        self.clear_since = None
        self.alert_active = False

    def update(self, unknown_count):
        current_time = time.monotonic()

        if unknown_count > 0:
            self.clear_since = None

            if self.unknown_since is None:
                self.unknown_since = current_time

            if current_time - self.unknown_since >= self.trigger_delay:
                self.alert_active = True

        else:
            self.unknown_since = None

            if self.alert_active:
                if self.clear_since is None:
                    self.clear_since = current_time

                if current_time - self.clear_since >= self.clear_delay:
                    self.alert_active = False
                    self.clear_since = None
            else:
                self.clear_since = None

        return self.alert_active
