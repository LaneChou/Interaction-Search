import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from collision_detection import ObstaclePath, detect_predicted_route_collision, path_cumsum


@dataclass
class PlannedTrajectory:
    xs: np.ndarray
    ys: np.ndarray
    yaws: np.ndarray
    vs: np.ndarray
    t_axis: np.ndarray
    source: str
    goal_xy: Tuple[float, float]


def smoothstep(z):
    z = np.clip(z, 0.0, 1.0)
    return z * z * (3.0 - 2.0 * z)


def retime_follow_to_obs_speed(
    plan: PlannedTrajectory,
    v_obs: float,
    contact_idx: int,
    margin_time: float = 0.6,
    a_dec_max: float = 3.0,
    min_speed: float = 0.0,
) -> PlannedTrajectory:
    xs = plan.xs
    ys = plan.ys
    vs = plan.vs
    t_axis = plan.t_axis
    n = len(xs)
    if n < 3:
        return plan

    s = np.zeros(n)
    for i in range(1, n):
        s[i] = s[i - 1] + math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])

    dt_nom = float(np.mean(np.diff(t_axis))) if len(t_axis) > 1 else 1.0 / 60.0
    contact_idx = int(np.clip(contact_idx, 1, n - 1))
    v_start = float(max(vs[0], min_speed))
    v_contact = float(min(v_start, max(v_obs, min_speed)))
    v_cap = vs.copy()

    for k in range(contact_idx + 1):
        z = k / max(contact_idx, 1)
        sigma = smoothstep(z)
        v_lim = v_start + (v_contact - v_start) * sigma
        v_cap[k] = min(v_cap[k], v_lim)

    v_cap[: contact_idx + 1] = np.minimum.accumulate(v_cap[: contact_idx + 1])
    v_cap[contact_idx:] = np.minimum(v_cap[contact_idx:], v_contact)
    v_cap = np.maximum(v_cap, min_speed)

    v_adj = v_cap.copy()
    for i in range(1, contact_idx + 1):
        ds = s[i] - s[i - 1]
        v_mid = max(0.5 * (v_adj[i] + v_adj[i - 1]), 0.1)
        dt_i = ds / v_mid if v_mid > 1e-6 else dt_nom
        v_floor = max(min_speed, v_adj[i - 1] - a_dec_max * dt_i)
        v_adj[i] = max(v_adj[i], v_floor)
        v_adj[i] = min(v_adj[i], v_adj[i - 1])

    for i in range(contact_idx + 1, n):
        v_adj[i] = min(v_adj[i], v_contact)

    t_new = np.zeros(n)
    for i in range(1, n):
        ds = s[i] - s[i - 1]
        v_mid = max(0.5 * (v_adj[i] + v_adj[i - 1]), 0.1)
        t_new[i] = t_new[i - 1] + ds / v_mid

    return PlannedTrajectory(xs.copy(), ys.copy(), plan.yaws.copy(), v_adj, t_new, "retimed", plan.goal_xy)


def truncate_plan_to_contact(plan: PlannedTrajectory, contact_idx: int) -> PlannedTrajectory:
    contact_idx = int(np.clip(contact_idx, 0, len(plan.xs) - 1))
    end = contact_idx + 1
    return PlannedTrajectory(
        xs=plan.xs[:end].copy(),
        ys=plan.ys[:end].copy(),
        yaws=plan.yaws[:end].copy(),
        vs=plan.vs[:end].copy(),
        t_axis=plan.t_axis[:end].copy(),
        source=plan.source,
        goal_xy=plan.goal_xy,
    )


def demo_collision_and_retime():
    dt = 0.2
    t_axis = np.arange(0.0, 5.0 + 1e-9, dt)
    xs = 4.0 * t_axis
    ys = np.zeros_like(xs)
    yaws = np.zeros_like(xs)
    vs = np.full_like(xs, 4.0)
    plan = PlannedTrajectory(xs, ys, yaws, vs, t_axis, "demo_seed", (float(xs[-1]), float(ys[-1])))

    obstacle_route = np.array([[6.0, -5.0], [6.0, 5.0]], dtype=float)
    obstacle = ObstaclePath(
        coords=obstacle_route,
        cum_s=path_cumsum(obstacle_route),
        length=10.0,
        s0=2.5,
        v=1.0,
        yaw=math.pi / 2.0,
        actor_id=1,
        role_name="demo_cross_vehicle",
    )

    has_collision, contact_idx, contact_pt, v_proj, obs = detect_predicted_route_collision(
        plan,
        [obstacle],
        radius=1.5,
        time_horizon=2.0,
        back_time=0.8,
        obs_dt=0.1,
    )

    print(f"has_collision={has_collision}")
    if not has_collision:
        return plan

    print(f"contact_idx={contact_idx}, contact_pt={contact_pt}, v_proj={v_proj:.2f}, obstacle={obs.role_name}")
    retimed = retime_follow_to_obs_speed(plan, v_obs=max(v_proj, 0.75), contact_idx=contact_idx, min_speed=0.75)
    visual = truncate_plan_to_contact(retimed, contact_idx)
    print("first 8 seed speeds:", np.round(plan.vs[:8], 2).tolist())
    print("first 8 retimed speeds:", np.round(retimed.vs[:8], 2).tolist())
    print("first 8 retimed times:", np.round(retimed.t_axis[:8], 2).tolist())
    print(f"visual trajectory points before contact: {len(visual.xs)}")
    return retimed


if __name__ == "__main__":
    demo_collision_and_retime()
