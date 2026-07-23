from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from hoverpilot.rflink.models import FlightAxisState


ACTION_LABELS = ("ail", "ele", "thr", "rud")


def format_debug_state(debug_state: Optional[Mapping[str, object]]) -> str:
    if not isinstance(debug_state, Mapping):
        return "debug unavailable"

    return (
        "DBG "
        f"x={float(debug_state.get('x_m', 0.0)):8.2f} "
        f"y={float(debug_state.get('y_m', 0.0)):8.2f} "
        f"agl={float(debug_state.get('altitude_agl_m', 0.0)):6.3f} "
        f"inc={float(debug_state.get('inclination_deg', 0.0)):+6.2f} "
        f"q={float(debug_state.get('pitch_rate_deg_s', 0.0)):+7.2f} "
        f"vel=({float(debug_state.get('velocity_world_u_mps', 0.0)):+5.2f},"
        f"{float(debug_state.get('velocity_world_v_mps', 0.0)):+5.2f},"
        f"{float(debug_state.get('velocity_world_w_mps', 0.0)):+5.2f}) "
        f"lost={float(debug_state.get('lost_components', 0.0)):0.1f} "
        f"locked={float(debug_state.get('vehicle_locked', 0.0)):0.1f} "
        f"ground={float(debug_state.get('touching_ground', 0.0)):0.1f} "
        f"ground_t={float(debug_state.get('ground_contact_duration_s', 0.0)):4.2f} "
        f"engine={float(debug_state.get('engine_running', 0.0)):0.1f} "
        f"ctrl={float(debug_state.get('controller_active', 0.0)):0.1f} "
        f"reset_btn={float(debug_state.get('reset_button_pressed', 0.0)):0.1f} "
        f"t={float(debug_state.get('physics_time_s', 0.0)):8.3f} "
        f"status={float(debug_state.get('aircraft_status', 0.0)):0.1f}"
    )


def format_action(action: Sequence[float]) -> str:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.shape != (4,):
        raise ValueError("action must have shape (4,)")
    return "TX " + " ".join(
        f"{label}={value:+0.3f}" for label, value in zip(ACTION_LABELS, values)
    )


def format_state(state: FlightAxisState) -> str:
    return (
        "RX "
        f"t={state.m_currentPhysicsTime_SEC:8.3f}s "
        f"pos=({state.m_aircraftPositionX_MTR:8.2f}, {state.m_aircraftPositionY_MTR:8.2f})m "
        f"agl={state.m_altitudeAGL_MTR:6.3f}m "
        f"air={state.m_airspeed_MPS:5.2f}m/s "
        f"gs={state.m_groundspeed_MPS:5.2f}m/s "
        f"att=(az={state.m_azimuth_DEG:6.1f}, inc={state.m_inclination_DEG:6.1f}, roll={state.m_roll_DEG:6.1f}) "
        f"rates=(p={state.m_pitchRate_DEGpSEC:6.1f}, r={state.m_rollRate_DEGpSEC:6.1f}, y={state.m_yawRate_DEGpSEC:6.1f})"
    )


def format_step_log(
    *,
    action: Sequence[float],
    info: Mapping[str, object],
    reward: float,
    terminated: bool,
    truncated: bool,
) -> str:
    debug_state = info.get("debug_state", {})
    if not isinstance(debug_state, Mapping):
        debug_state = {}

    state_summary = info.get("state_summary")
    if isinstance(state_summary, str):
        state_text = state_summary
    else:
        state_text = (
            f"agl={float(debug_state.get('altitude_agl_m', 0.0)):0.3f} "
            f"x={float(debug_state.get('x_m', 0.0)):0.2f} "
            f"y={float(debug_state.get('y_m', 0.0)):0.2f}"
        )

    return (
        f"{format_action(action)} | "
        f"reward={reward:+0.3f} terminated={terminated} truncated={truncated} "
        f"reason={info.get('termination_reason')} | "
        f"{state_text}"
    )
