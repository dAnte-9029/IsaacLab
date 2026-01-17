"""Phase-warp utilities for non-uniform flapping cycles."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PhaseWarpState:
    """Instantaneous phase and its time derivatives."""

    psi: float
    psi_dot: float
    psi_ddot: float
    psi_dddot: float


class PhaseWarp:
    """Phase-warp a 2π cycle so downstroke takes delta*T and upstroke (1-delta)*T."""

    def __init__(
        self,
        *,
        f_hz: float,
        delta: float,
        smoothness: float,
        mode: str = "fixed_f",
        f_ref_hz: float = 4.5,
    ) -> None:
        if f_hz <= 0.0:
            raise ValueError("f_hz must be positive.")
        if not (0.0 < delta < 1.0):
            raise ValueError("downstroke ratio delta must be in (0,1).")
        if f_ref_hz <= 0.0:
            raise ValueError("f_ref_hz must be positive.")
        mode = str(mode)
        if mode not in ("fixed_f", "fixed_peak_qd"):
            raise ValueError("mode must be 'fixed_f' or 'fixed_peak_qd'.")

        if mode == "fixed_peak_qd":
            f_eff = 2.0 * delta * f_ref_hz
        else:
            f_eff = f_hz
        if f_eff <= 0.0:
            raise ValueError("Computed f_eff must be positive.")

        self.f_eff = float(f_eff)
        self.delta = float(delta)
        self.T = 1.0 / float(f_eff)

        self.f_down = float(self.f_eff / (2.0 * self.delta))
        self.f_up = float(self.f_eff / (2.0 * (1.0 - self.delta)))

        s_max = 2.0 * min(self.delta, 1.0 - self.delta)
        s = max(0.0, min(float(smoothness), s_max))
        self.s = s
        self.u0 = self.delta - 0.5 * s
        self.u1 = self.delta + 0.5 * s

        self.r_down = 0.5 / self.delta
        self.r_up = 0.5 / (1.0 - self.delta)

    def _smoothstep(self, x: float) -> float:
        return x * x * (3.0 - 2.0 * x)

    def _smoothstep_d1(self, x: float) -> float:
        return 6.0 * x - 6.0 * x * x

    def _smoothstep_d2(self, x: float) -> float:
        return 6.0 - 12.0 * x

    def eval(self, t: float) -> PhaseWarpState:
        """Return phase and its time derivatives at time t."""
        if self.s <= 0.0:
            u = (t % self.T) / self.T
            if u <= self.delta:
                p = self.r_down * u
                p_d = self.r_down
                p_dd = 0.0
                p_ddd = 0.0
            else:
                p = 0.5 + self.r_up * (u - self.delta)
                p_d = self.r_up
                p_dd = 0.0
                p_ddd = 0.0
        else:
            u = (t % self.T) / self.T
            if u <= self.u0:
                p = self.r_down * u
                p_d = self.r_down
                p_dd = 0.0
                p_ddd = 0.0
            elif u >= self.u1:
                p_u1 = self._p_at_u1()
                p = p_u1 + self.r_up * (u - self.u1)
                p_d = self.r_up
                p_dd = 0.0
                p_ddd = 0.0
            else:
                x = (u - self.u0) / self.s
                ss = self._smoothstep(x)
                p = self._p_at_u0() + self.r_down * (u - self.u0) + (self.r_up - self.r_down) * self.s * (
                    x**3 - 0.5 * x**4
                )
                p_d = self.r_down + (self.r_up - self.r_down) * ss
                p_dd = (self.r_up - self.r_down) * self._smoothstep_d1(x) / self.s
                p_ddd = (self.r_up - self.r_down) * self._smoothstep_d2(x) / (self.s * self.s)

        w = 2.0 * math.pi / self.T
        psi = 2.0 * math.pi * p
        psi_dot = w * p_d
        psi_ddot = (2.0 * math.pi / (self.T * self.T)) * p_dd
        psi_dddot = (2.0 * math.pi / (self.T * self.T * self.T)) * p_ddd
        return PhaseWarpState(psi=psi, psi_dot=psi_dot, psi_ddot=psi_ddot, psi_dddot=psi_dddot)

    def _p_at_u0(self) -> float:
        return self.r_down * self.u0

    def _p_at_u1(self) -> float:
        return self._p_at_u0() + 0.5 * self.s * (self.r_down + self.r_up)

    def cos_kinematics(self, t: float, amp: float) -> tuple[float, float, float, float]:
        """Return q, qd, qdd, qddd for q=amp*cos(psi(t))."""
        st = self.eval(t)
        s = math.sin(st.psi)
        c = math.cos(st.psi)
        q = amp * c
        qd = -amp * s * st.psi_dot
        qdd = -amp * (c * (st.psi_dot**2) + s * st.psi_ddot)
        qddd = amp * (s * (st.psi_dot**3) - 3.0 * c * st.psi_dot * st.psi_ddot - s * st.psi_dddot)
        return q, qd, qdd, qddd
