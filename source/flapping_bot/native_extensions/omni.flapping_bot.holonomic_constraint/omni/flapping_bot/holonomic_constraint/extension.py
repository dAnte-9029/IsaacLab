"""Kit lifecycle for the native flapping-wing trajectory constraint."""

from __future__ import annotations

from pathlib import Path

import omni.ext
from pxr import Plug


class HolonomicConstraintExtension(omni.ext.IExt):
    """Register and unregister the native PhysX custom joint."""

    def on_startup(self, ext_id: str) -> None:
        del ext_id
        resources_dir = Path(__file__).resolve().parents[3] / "resources"
        plugins = Plug.Registry().RegisterPlugins(str(resources_dir))
        if not plugins:
            raise RuntimeError(f"Failed to register USD schema resources from {resources_dir}.")

        from . import _native

        if not _native.initialize():
            raise RuntimeError("Failed to register FlappingWingTrajectoryJoint.")

    def on_shutdown(self) -> None:
        from . import _native

        _native.shutdown()
