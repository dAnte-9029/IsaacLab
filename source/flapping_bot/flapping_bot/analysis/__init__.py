"""Deterministic analysis workflows for the flapping-bot project."""

__all__ = ["TailAuditSettings", "run_tail_unit_audit"]


def __getattr__(name: str) -> object:
    if name in __all__:
        from .tail_unit_audit import TailAuditSettings, run_tail_unit_audit

        return {
            "TailAuditSettings": TailAuditSettings,
            "run_tail_unit_audit": run_tail_unit_audit,
        }[name]
    raise AttributeError(name)
