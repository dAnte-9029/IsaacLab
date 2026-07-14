"""Single headless Isaac application shared by DeLaurier integration tests."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app
