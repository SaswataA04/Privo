
import os
import sys


class StartupManager:

    def __init__(self):

        # Determine the project directory.
        self.base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        # Windows Startup folder.
        self.startup_folder = os.path.join(
            os.environ["APPDATA"],
            "Microsoft",
            "Windows",
            "Start Menu",
            "Programs",
            "Startup"
        )

        # Launcher filename.
        self.launcher_name = "PrivacyGuard.bat"

        self.launcher_path = os.path.join(
            self.startup_folder,
            self.launcher_name
        )

    def is_enabled(self):

        """Check whether Privacy Guard is registered at startup."""

        return os.path.isfile(self.launcher_path)

    def enable(self):

        """Create a launcher in the Windows Startup folder."""

        os.makedirs(
            self.startup_folder,
            exist_ok=True
        )

        # Use the same Python interpreter running this application.
        python_executable = sys.executable

        main_file = os.path.join(
            self.base_dir,
            "main.py"
        )

        # Build the batch launcher.
        launcher_content = (
            "@echo off\n"
            f'cd /d "{self.base_dir}"\n'
            f'"{python_executable}" "{main_file}" --minimized\n'
        )

        with open(
            self.launcher_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(launcher_content)

        return True

    def disable(self):

        """Remove the startup launcher."""

        if os.path.isfile(self.launcher_path):

            os.remove(self.launcher_path)

        return True
