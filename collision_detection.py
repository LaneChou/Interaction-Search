import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ObstaclePath:
    coords: np.ndarray
    cum_s: np.ndarray
    length: float
    s0: float
    v: float
    yaw: float = 0.0
    actor_id: int = -1
    role_name: str = "obstacle"


def path_cumsum(coords: np.ndarray) -> np.ndarray:
    if len(coords) < 2:
        return np.array([0.0], dtype=float)
    seg = np.diff(coords, axis=0)
    d = np.sqrt(np.sum(seg * seg, axis=1))
    return np.concatenate([[0.0], np.cumsum(d)])


def path_interp(coords: np.ndarray, cum_s: np.ndarray, s_query: float) -> np.ndarray:
    if len(coords) == 0:
        raise ValueError("coords must not be empty")
    if len(coords) == 1:
        return coords[0].astype(float)

    s_query = float(np.clip(s_query, 0.0, cum_s[-1]))
    j = int(np.searchsorted(cum_s, s_query, side="right")) - 1
    j = int(np.clip(j, 0, len(cum_s) - 2))
    denom = max(cum_s[j + 1] - cum_s[j], 1e-9)
    t = (s_query - cum_s[j]) / denom
    return coords[j] * (1.0 - t) + coords[j + 1] * t


def unit_tangent(xs: np.ndarray, ys: np.ndarray, i: int) -> np.ndarray:
    n = len(xs)
    if n < 2:
        return np.array([1.0, 0.0], dtype=float)
    if i <= 0:
        dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    elif i >= n - 1:
        dx, dy = xs[-1] - xs[-2], ys[-1] - ys[-2]
    else:
        dx, dy = xs[i + 1] - xs[i - 1], ys[i + 1] - ys[i - 1]
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=float)
    return np.array([dx / norm, dy / norm], dtype=float)


def detect_predicted_route_collision(
    plan,
    obstacles: List[ObstaclePath],
    radius: float,
    time_horizon: float,
    back_time: float = 0.8,
    obs_dt: float = 0.1,
) -> Tuple[bool, Optional[int], Optional[Tuple[float, float]], Optional[float], Optional[ObstaclePath]]:
    if len(plan.xs) == 0 or not obstacles:
        return False, None, None, None, None

    t_grid = np.arange(-max(0.0, back_time), max(0.0, time_horizon) + 1e-9, max(obs_dt, 1e-3))
    if len(t_grid) == 0:
        t_grid = np.array([0.0], dtype=float)

    r2 = radius * radius
    obstacle_walls = []
    for obs in obstacles:
        s_grid = np.clip(obs.s0 + obs.v * t_grid, 0.0, obs.length)
        ox = np.empty_like(s_grid)
        oy = np.empty_like(s_grid)
        for j, s_val in enumerate(s_grid):
            p_obs = path_interp(obs.coords, obs.cum_s, float(s_val))
            ox[j] = float(p_obs[0])
            oy[j] = float(p_obs[1])
        obstacle_walls.append((obs, s_grid, ox, oy))

    for i in range(len(plan.xs)):
        ex = float(plan.xs[i])
        ey = float(plan.ys[i])
        ego_tangent = unit_tangent(plan.xs, plan.ys, i)
        for obs, s_grid, ox, oy in obstacle_walls:
            dx = ex - ox
            dy = ey - oy
            d2 = dx * dx + dy * dy
            j_min = int(np.argmin(d2))
            if float(d2[j_min]) <= r2:
                s_here = float(s_grid[j_min])
                ds = max(0.05, 0.02 * obs.length)
                s0 = max(0.0, s_here - ds)
                s1 = min(obs.length, s_here + ds)
                p0 = path_interp(obs.coords, obs.cum_s, s0)
                p1 = path_interp(obs.coords, obs.cum_s, s1)
                obs_dir = p1 - p0
                obs_dir_norm = float(np.linalg.norm(obs_dir))
                if obs_dir_norm < 1e-9:
                    v_proj = 0.0
                else:
                    v_obs_vec = (obs_dir / obs_dir_norm) * obs.v
                    v_proj = float(np.dot(v_obs_vec, ego_tangent))
                return True, i, (float(ox[j_min]), float(oy[j_min])), max(v_proj, 0.0), obs

    return False, None, None, None, None
