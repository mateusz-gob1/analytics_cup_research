import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from matplotlib import animation as mpl_animation
from matplotlib import colors as mcolors
from matplotlib.collections import PolyCollection

try:
    from scipy.signal import savgol_filter
    _HAS_SG = True
except Exception:
    _HAS_SG = False


def _wrap_360(deg):
    return (deg + 360.0) % 360.0


def _wrap_180(deg):
    d = _wrap_360(deg)
    return np.where(d > 180.0, d - 360.0, d)


def _bounded01(x):
    return float(np.clip(x, 0.0, 1.0))


def _player_bases_from_team_df(tracking_team_df: pd.DataFrame) -> list[str]:
    bases_x = {c[:-2] for c in tracking_team_df.columns if isinstance(c, str) and c.endswith("_x")}
    bases_y = {c[:-2] for c in tracking_team_df.columns if isinstance(c, str) and c.endswith("_y")}
    bases = bases_x & bases_y
    bases = [b for b in bases if str(b).isdigit()]
    return sorted(bases, key=lambda b: int(b))


def _player_bases(df: pd.DataFrame) -> list[str]:
    bx = {c[:-2] for c in df.columns if isinstance(c, str) and c.endswith("_x")}
    by = {c[:-2] for c in df.columns if isinstance(c, str) and c.endswith("_y")}
    bases = [b for b in (bx & by) if str(b).isdigit()]
    return sorted(bases, key=lambda b: int(b))


def _player_bases_only_numeric(df: pd.DataFrame) -> list[str]:
    bx = {c[:-2] for c in df.columns if isinstance(c, str) and c.endswith("_x")}
    by = {c[:-2] for c in df.columns if isinstance(c, str) and c.endswith("_y")}
    bases = [b for b in (bx & by) if b.isdigit()]
    return sorted(bases, key=lambda s: int(s))


def _coords_for_frame(row: pd.Series, bases: list[str]) -> tuple[np.ndarray, list[int]]:
    pts = []
    pids = []
    for b in bases:
        x = row.get(f"{b}_x")
        y = row.get(f"{b}_y")
        if pd.notna(x) and pd.notna(y):
            pts.append([float(x), float(y)])
            pids.append(int(b))
    if pts:
        return np.array(pts, dtype=float), pids
    return np.empty((0, 2), dtype=float), []


def _to_mpl(xy: np.ndarray, L: float, W: float) -> np.ndarray:
    out = xy.copy()
    out[:, 0] += L / 2.0
    out[:, 1] += W / 2.0
    return out


def _smooth_series(s: pd.Series, method: str, window: int, poly: int) -> pd.Series:
    if method == "sg" and _HAS_SG:
        arr = s.astype(float).interpolate(limit_direction="both").to_numpy()
        w = min(window, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
        if w < 3:
            return s.astype(float).interpolate(limit_direction="both")
        try:
            sm = savgol_filter(arr, window_length=w, polyorder=min(poly, w - 1), mode="interp")
            return pd.Series(sm, index=s.index)
        except Exception:
            return s.astype(float).interpolate(limit_direction="both").ewm(span=max(2, window // 2), adjust=False).mean()
    if method == "ema":
        return s.astype(float).interpolate(limit_direction="both").ewm(span=max(2, window), adjust=False).mean()
    return s.astype(float)


def _to_timedelta(val):
    if isinstance(val, pd.Timedelta):
        return val
    if isinstance(val, str):
        if ":" in val:
            return pd.to_timedelta(val)
        parts = val.split(".")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            minutes = int(parts[0])
            seconds = float(parts[1])
            return pd.to_timedelta(minutes * 60 + seconds, unit="s")
        return pd.to_timedelta(val)
    if val is None:
        return None
    return pd.to_timedelta(val, unit="s")


def _resolve_frame_id(
    tracking_df: pd.DataFrame,
    frame_id: int,
    required_frames: set[int] | None = None
) -> int:
    if "frame_id" not in tracking_df.columns:
        raise ValueError("tracking_df must contain 'frame_id' column")
    frames = pd.Series(tracking_df["frame_id"].astype(int).unique())
    if required_frames is not None:
        frames = frames[frames.isin(list(required_frames))]
    if frames.empty:
        raise ValueError("No available frame_id to choose from.")
    candidates = frames[frames <= int(frame_id)]
    if candidates.empty:
        raise ValueError(f"No frame_id <= {frame_id} in data.")
    return int(candidates.max())


def _frame_list_from_inputs(
    tracking_df: pd.DataFrame,
    frame_ids: list[int] | None = None,
    start_time=None,
    end_time=None
) -> list[int]:
    if frame_ids is None:
        if start_time is not None or end_time is not None:
            frame_ids = frame_ids_from_time_range(tracking_df, start_time, end_time)
        else:
            frame_ids = tracking_df["frame_id"].astype(int).tolist()
    else:
        frame_ids = [int(fid) for fid in frame_ids]
    return list(dict.fromkeys(frame_ids))


def build_delaunay_per_frame(
    tracking_team_df: pd.DataFrame,
    min_points: int = 3
) -> dict[int, list[tuple[int, int, int]]]:
    if "frame_id" not in tracking_team_df.columns:
        raise ValueError("tracking_team_df must contain 'frame_id' column")

    bases = _player_bases_from_team_df(tracking_team_df)
    triangles_by_frame: dict[int, list[tuple[int, int, int]]] = {}

    for _, row in tracking_team_df.sort_values("frame_id").iterrows():
        frame = int(row["frame_id"])
        pts, pids = _coords_for_frame(row, bases)

        if pts.shape[0] < min_points:
            triangles_by_frame[frame] = []
            continue

        try:
            tri = Delaunay(pts)
            simplices = tri.simplices
            triangles = [tuple(sorted((pids[i], pids[j], pids[k]))) for i, j, k in simplices]
            triangles_by_frame[frame] = list({t for t in triangles})
        except QhullError:
            triangles_by_frame[frame] = []

    return triangles_by_frame


def plot_team_triangulation_on_frame(
    tracking_team_df: pd.DataFrame,
    triangles_by_frame: dict[int, list[tuple[int, int, int]]],
    frame_id: int,
    pitch_length: float = 104,
    pitch_width: float = 68,
    annotate_ids: bool = True,
    opponent_df: pd.DataFrame | None = None,
    show_opponents: bool = True,
    figsize=(10, 6)
):
    pitch = Pitch(pitch_length=pitch_length, pitch_width=pitch_width)
    fig, ax = pitch.draw(figsize=figsize)

    frame_id = _resolve_frame_id(tracking_team_df, frame_id, required_frames=set(triangles_by_frame.keys()))
    row = tracking_team_df.loc[tracking_team_df["frame_id"] == frame_id].iloc[0]

    bases = _player_bases_from_team_df(tracking_team_df)
    pts, pids = _coords_for_frame(row, bases)
    if pts.size == 0:
        ax.set_title(f"Frame {frame_id}: brak punktow")
        return fig, ax

    pts_plot = pts.copy()
    pts_plot[:, 0] += pitch_length / 2
    pts_plot[:, 1] += pitch_width / 2

    pitch.scatter(pts_plot[:, 0], pts_plot[:, 1], ax=ax, s=220, marker="o",
                  facecolor="#1f77b4", edgecolor="black", label="Team")
    if annotate_ids:
        for (x, y), pid in zip(pts_plot, pids):
            ax.text(x, y, str(pid), ha="center", va="center", fontsize=8)

    if opponent_df is not None and show_opponents:
        opp_row = opponent_df.loc[opponent_df["frame_id"] == frame_id]
        if not opp_row.empty:
            opp_row = opp_row.iloc[0]
            opp_bases = _player_bases_from_team_df(opponent_df)
            opp_pts, _ = _coords_for_frame(opp_row, opp_bases)
            if opp_pts.size:
                opp_plot = opp_pts.copy()
                opp_plot[:, 0] += pitch_length / 2
                opp_plot[:, 1] += pitch_width / 2
                pitch.scatter(opp_plot[:, 0], opp_plot[:, 1], ax=ax, s=180, marker="o",
                              facecolor="#d62728", edgecolor="black", label="Opponent")

    tris = triangles_by_frame.get(frame_id, [])
    for (a, b, c) in tris:
        try:
            ia = pids.index(a); ib = pids.index(b); ic = pids.index(c)
        except ValueError:
            continue
        xa, ya = pts_plot[ia]; xb, yb = pts_plot[ib]; xc, yc = pts_plot[ic]
        ax.plot([xa, xb], [ya, yb], linewidth=1.5)
        ax.plot([xb, xc], [yb, yc], linewidth=1.5)
        ax.plot([xc, xa], [yc, ya], linewidth=1.5)

    ax.legend(loc="upper center", ncol=3, frameon=False)
    ax.set_title(f"Frame {frame_id}: triangulacja Delaunay (team)")
    return fig, ax


def compute_player_kinematics(
    tracking_df: pd.DataFrame,
    fps: float,
    smooth: str = "sg",
    window: int = 7,
    poly: int = 2,
    interpolate_gaps: bool = True,
    max_gap: int = 5,
    direction_min_speed: float = 0.3,
    compute_ball: bool = True
) -> pd.DataFrame:
    if "frame_id" not in tracking_df.columns:
        raise ValueError("tracking_df must contain 'frame_id'")
    out = tracking_df.copy().sort_values("frame_id").reset_index(drop=True)
    dt = 1.0 / float(fps)
    bases = _player_bases(out)

    if compute_ball:
        if not {"ball_x", "ball_y"}.issubset(out.columns):
            raise ValueError("Missing ball_x/ball_y")
        bx = out["ball_x"].astype(float)
        by = out["ball_y"].astype(float)
        if interpolate_gaps:
            bx = bx.interpolate(limit=max_gap, limit_direction="both")
            by = by.interpolate(limit=max_gap, limit_direction="both")
        if smooth in ("sg", "ema"):
            bx_s = _smooth_series(bx, smooth, window, poly)
            by_s = _smooth_series(by, smooth, window, poly)
        else:
            bx_s, by_s = bx, by
        ball_vx = np.gradient(bx_s.to_numpy(), dt)
        ball_vy = np.gradient(by_s.to_numpy(), dt)
        ball_speed = np.hypot(ball_vx, ball_vy)
        ball_dir = np.degrees(np.arctan2(ball_vy, ball_vx))
        ball_dir = np.where(ball_speed >= direction_min_speed, _wrap_360(ball_dir), np.nan)
        out["ball_vx"] = ball_vx
        out["ball_vy"] = ball_vy
        out["ball_speed"] = ball_speed
        out["ball_dir"] = ball_dir

    new_cols: dict[str, np.ndarray] = {}

    for b in bases:
        x = out[f"{b}_x"].astype(float)
        y = out[f"{b}_y"].astype(float)

        if interpolate_gaps:
            x = x.interpolate(limit=max_gap, limit_direction="both")
            y = y.interpolate(limit=max_gap, limit_direction="both")

        if smooth in ("sg", "ema"):
            x_s = _smooth_series(x, smooth, window, poly)
            y_s = _smooth_series(y, smooth, window, poly)
        else:
            x_s, y_s = x, y

        vx = np.gradient(x_s.to_numpy(), dt)
        vy = np.gradient(y_s.to_numpy(), dt)

        if smooth == "ema":
            vx = pd.Series(vx).ewm(span=max(2, window), adjust=False).mean().to_numpy()
            vy = pd.Series(vy).ewm(span=max(2, window), adjust=False).mean().to_numpy()
        elif smooth == "sg" and _HAS_SG and len(vx) >= 5:
            wv = max(5, min(window, len(vx) if len(vx) % 2 == 1 else len(vx) - 1))
            try:
                vx = savgol_filter(vx, window_length=wv, polyorder=min(2, wv - 1), mode="interp")
                vy = savgol_filter(vy, window_length=wv, polyorder=min(2, wv - 1), mode="interp")
            except Exception:
                pass

        speed = np.hypot(vx, vy)
        dir_abs = np.degrees(np.arctan2(vy, vx))
        dir_abs = np.where(speed >= direction_min_speed, _wrap_360(dir_abs), np.nan)

        ax = np.gradient(vx, dt)
        ay = np.gradient(vy, dt)
        if smooth == "ema":
            ax = pd.Series(ax).ewm(span=max(2, window), adjust=False).mean().to_numpy()
            ay = pd.Series(ay).ewm(span=max(2, window), adjust=False).mean().to_numpy()
        elif smooth == "sg" and _HAS_SG and len(ax) >= 5:
            wa = max(5, min(window, len(ax) if len(ax) % 2 == 1 else len(ax) - 1))
            try:
                ax = savgol_filter(ax, window_length=wa, polyorder=min(2, wa - 1), mode="interp")
                ay = savgol_filter(ay, window_length=wa, polyorder=min(2, wa - 1), mode="interp")
            except Exception:
                pass

        dax = np.gradient(ax, dt)
        day = np.gradient(ay, dt)
        if smooth == "ema":
            dax = pd.Series(dax).ewm(span=max(2, window), adjust=False).mean().to_numpy()
            day = pd.Series(day).ewm(span=max(2, window), adjust=False).mean().to_numpy()
        elif smooth == "sg" and _HAS_SG and len(dax) >= 5:
            wd = max(5, min(window, len(dax) if len(dax) % 2 == 1 else len(dax) - 1))
            try:
                dax = savgol_filter(dax, window_length=wd, polyorder=min(2, wd - 1), mode="interp")
                day = savgol_filter(day, window_length=wd, polyorder=min(2, wd - 1), mode="interp")
            except Exception:
                pass
        dacc = np.hypot(dax, day)

        new_cols[f"{b}_vx"] = vx
        new_cols[f"{b}_vy"] = vy
        new_cols[f"{b}_speed"] = speed
        new_cols[f"{b}_ax"] = ax
        new_cols[f"{b}_ay"] = ay
        new_cols[f"{b}_dacc"] = dacc
        new_cols[f"{b}_dir"] = dir_abs

        if compute_ball and {"ball_x", "ball_y"}.issubset(out.columns):
            bearing = _wrap_360(np.degrees(np.arctan2(out["ball_y"] - y_s, out["ball_x"] - x_s)))
            new_cols[f"{b}_dir_to_ball"] = bearing
            if "ball_dir" in out.columns:
                rel = _wrap_360(dir_abs - out["ball_dir"])
                new_cols[f"{b}_dir_rel_ball"] = rel
                new_cols[f"{b}_dir_rel_ball180"] = _wrap_180(rel)
            aim = _wrap_360(dir_abs - bearing)
            new_cols[f"{b}_aim_error_to_ball"] = aim
            new_cols[f"{b}_aim_error_to_ball180"] = _wrap_180(aim)

    if new_cols:
        out = out.assign(**new_cols)

    return out


def kuramoto_R_deg(angles_deg, weights=None):
    ang = np.deg2rad(np.asarray(angles_deg))
    if weights is None:
        z = np.exp(1j * ang).mean()
    else:
        w = np.asarray(weights, float)
        z = (w * np.exp(1j * ang)).sum() / (w.sum() + 1e-9)
    return float(np.abs(z))


def triangle_dir_sync_relative_to_ball(row, tri, vmin=1.2, min_active=2):
    dirs, bears, speeds = [], [], []
    for pid in tri:
        d = row.get(f"{pid}_dir", np.nan)
        s = row.get(f"{pid}_speed", np.nan)
        x = row.get(f"{pid}_x", np.nan); y = row.get(f"{pid}_y", np.nan)
        bx = row.get("ball_x", np.nan); by = row.get("ball_y", np.nan)
        if not (np.isfinite(d) and np.isfinite(s) and np.isfinite(x) and np.isfinite(y) and np.isfinite(bx) and np.isfinite(by)):
            continue
        dirs.append(d); speeds.append(s)
        dx = bx - x; dy = by - y
        bears.append(_wrap_360(np.degrees(np.arctan2(dy, dx))))
    if len(dirs) < min_active:
        return np.nan, np.nan, np.nan, 0

    dirs = np.array(dirs, float)
    bears = np.array(bears, float)
    speeds = np.array(speeds, float)

    active = speeds >= vmin
    if active.sum() < min_active:
        return np.nan, np.nan, np.nan, active.sum()

    phi = _wrap_360(dirs[active] - bears[active])
    w = speeds[active] / (speeds[active].sum() + 1e-9)
    R_dir = kuramoto_R_deg(phi, weights=w)
    align = float(np.mean(np.cos(np.deg2rad(phi))))
    mean_phi = float(((phi.mean() + 180) % 360) - 180)
    return R_dir, align, mean_phi, int(active.sum())


def sync_speed_dacc(row, tri, eps=1e-6):
    s_list, a_list = [], []
    for pid in tri:
        s = row.get(f"{pid}_speed", np.nan)
        a = row.get(f"{pid}_dacc", np.nan)
        if np.isfinite(s):
            s_list.append(float(s))
        if np.isfinite(a):
            a_list.append(float(a))

    S_speed = np.nan
    S_dacc = np.nan

    if len(s_list) >= 2:
        s = np.array(s_list)
        S_speed = 1.0 - (s.std(ddof=0) / (s.mean() + eps))
        S_speed = _bounded01(S_speed)

    if len(a_list) >= 2:
        a = np.array(a_list)
        med = np.median(a)
        mad = np.median(np.abs(a - med)) + eps
        cvr = mad / (np.abs(med) + mad)
        S_dacc = 1.0 - float(np.clip(cvr, 0, 1))

    return S_speed, S_dacc, len(s_list), len(a_list)


def triangle_sync_metric(row, tri, w_dir=0.5, w_spd=0.3, w_dacc=0.2, vmin=1.2):
    R_dir, align, mean_phi, n_dir = triangle_dir_sync_relative_to_ball(row, tri, vmin=vmin)
    S_speed, S_dacc, n_spd, n_dacc = sync_speed_dacc(row, tri)

    wD = w_dir if np.isfinite(R_dir) else 0.0
    wS = w_spd if np.isfinite(S_speed) else 0.0
    wA = w_dacc if np.isfinite(S_dacc) else 0.0
    W = wD + wS + wA

    if W > 0:
        Sync = (wD * (0 if not np.isfinite(R_dir) else R_dir) +
                wS * (0 if not np.isfinite(S_speed) else S_speed) +
                wA * (0 if not np.isfinite(S_dacc) else S_dacc)) / W
    else:
        Sync = np.nan

    return {
        "R_dir": R_dir, "align": align, "S_speed": S_speed, "S_dacc": S_dacc, "Sync": Sync,
        "n_dir": n_dir, "n_spd": n_spd, "n_dacc": n_dacc, "w_eff": W
    }


def compute_triangle_synchrony(
    df: pd.DataFrame,
    triangles_by_frame: dict[int, list[tuple[int, int, int]]],
    w_dir=0.5, w_spd=0.3, w_dacc=0.2,
    vmin=1.2
) -> pd.DataFrame:
    out = []
    frames_idx = df.set_index("frame_id")
    for frame_id, tris in triangles_by_frame.items():
        if frame_id not in frames_idx.index:
            continue
        row = frames_idx.loc[frame_id]
        for (p1, p2, p3) in tris:
            comps = triangle_sync_metric(row, (p1, p2, p3), w_dir, w_spd, w_dacc, vmin)
            out.append((
                frame_id, p1, p2, p3,
                comps["R_dir"], comps["align"], comps["S_speed"], comps["S_dacc"], comps["Sync"],
                comps["n_dir"], comps["n_spd"], comps["n_dacc"], comps["w_eff"]
            ))
    return pd.DataFrame(out, columns=[
        "frame_id", "pid1", "pid2", "pid3", "R_dir", "align", "S_speed", "S_dacc", "Sync",
        "n_dir", "n_spd", "n_dacc", "w_eff"
    ])


def team_sync_for_frame(
    sync_df: pd.DataFrame,
    frame_id: int,
    metric: str = "Sync",
    agg: str = "mean",
    min_triangles: int = 1
) -> float:
    frame = sync_df[sync_df["frame_id"] == frame_id]
    if frame.empty or len(frame) < min_triangles:
        return float("nan")
    values = frame[metric].astype(float)
    if agg == "mean":
        return float(np.nanmean(values.to_numpy()))
    if agg == "median":
        return float(np.nanmedian(values.to_numpy()))
    raise ValueError(f"Unknown agg: {agg}")


def team_sync_over_time(
    sync_df: pd.DataFrame,
    metric: str = "Sync",
    agg: str = "mean",
    min_triangles: int = 1
) -> pd.DataFrame:
    def _agg(group: pd.DataFrame) -> float:
        if len(group) < min_triangles:
            return float("nan")
        vals = group[metric].astype(float).to_numpy()
        if agg == "mean":
            return float(np.nanmean(vals))
        if agg == "median":
            return float(np.nanmedian(vals))
        raise ValueError(f"Unknown agg: {agg}")

    out = sync_df.groupby("frame_id", sort=True).apply(_agg)
    return out.reset_index(name="team_sync")


def frame_ids_from_time_range(
    tracking_df: pd.DataFrame,
    start_time=None,
    end_time=None
) -> list[int]:
    if "timestamp" not in tracking_df.columns:
        raise ValueError("tracking_df must contain 'timestamp' column")
    start_td = _to_timedelta(start_time)
    end_td = _to_timedelta(end_time)
    ts = tracking_df["timestamp"]
    mask = pd.Series(True, index=tracking_df.index)
    if start_td is not None:
        mask &= ts >= start_td
    if end_td is not None:
        mask &= ts <= end_td
    if "frame_id" not in tracking_df.columns:
        raise ValueError("tracking_df must contain 'frame_id' column")
    return tracking_df.loc[mask, "frame_id"].astype(int).tolist()


def filter_triangles_and_sync_by_time(
    tracking_df: pd.DataFrame,
    triangles_by_frame: dict[int, list[tuple[int, int, int]]],
    sync_df: pd.DataFrame,
    start_time=None,
    end_time=None
):
    frame_ids = set(frame_ids_from_time_range(tracking_df, start_time, end_time))
    triangles_filtered = {fid: tris for fid, tris in triangles_by_frame.items() if fid in frame_ids}
    sync_filtered = sync_df[sync_df["frame_id"].isin(frame_ids)].copy()
    return triangles_filtered, sync_filtered, sorted(frame_ids)


def compute_sync_for_time_range(
    tracking_df: pd.DataFrame,
    fps: float,
    start_time=None,
    end_time=None,
    buffer_frames: int = 5,
    smooth: str = "sg",
    window: int = 7,
    poly: int = 2,
    direction_min_speed: float = 0.3,
    compute_ball: bool = True,
    min_points: int = 3,
    w_dir: float = 0.5,
    w_spd: float = 0.3,
    w_dacc: float = 0.2,
    vmin: float = 1.2
):
    frame_ids = frame_ids_from_time_range(tracking_df, start_time, end_time)
    if not frame_ids:
        return {}, pd.DataFrame(), []
    min_f = min(frame_ids) - int(buffer_frames)
    max_f = max(frame_ids) + int(buffer_frames)
    df_buf = tracking_df[(tracking_df["frame_id"] >= min_f) & (tracking_df["frame_id"] <= max_f)].copy()

    triangles = build_delaunay_per_frame(df_buf, min_points=min_points)
    kin = compute_player_kinematics(
        df_buf, fps=fps, smooth=smooth, window=window, poly=poly,
        direction_min_speed=direction_min_speed, compute_ball=compute_ball
    )
    sync_df = compute_triangle_synchrony(
        kin, triangles, w_dir=w_dir, w_spd=w_spd, w_dacc=w_dacc, vmin=vmin
    )

    frame_set = set(frame_ids)
    triangles_filtered = {fid: tris for fid, tris in triangles.items() if fid in frame_set}
    sync_filtered = sync_df[sync_df["frame_id"].isin(frame_set)].copy()
    return triangles_filtered, sync_filtered, sorted(frame_set)


def group_sync_metric(
    row: pd.Series,
    pids,
    w_dir: float = 0.5,
    w_spd: float = 0.3,
    w_dacc: float = 0.2,
    vmin: float = 1.2,
    min_dir_active: int = 2
) -> dict:
    R_dir, align, mean_phi, n_dir = triangle_dir_sync_relative_to_ball(
        row, pids, vmin=vmin, min_active=min_dir_active
    )
    S_speed, S_dacc, n_spd, n_dacc = sync_speed_dacc(row, pids)

    wD = w_dir if np.isfinite(R_dir) else 0.0
    wS = w_spd if np.isfinite(S_speed) else 0.0
    wA = w_dacc if np.isfinite(S_dacc) else 0.0
    W = wD + wS + wA

    if W > 0:
        Sync = (wD * (0 if not np.isfinite(R_dir) else R_dir) +
                wS * (0 if not np.isfinite(S_speed) else S_speed) +
                wA * (0 if not np.isfinite(S_dacc) else S_dacc)) / W
    else:
        Sync = np.nan

    return {
        "R_dir": R_dir, "align": align, "S_speed": S_speed, "S_dacc": S_dacc, "Sync": Sync,
        "n_dir": n_dir, "n_spd": n_spd, "n_dacc": n_dacc, "w_eff": W
    }


def group_sync_for_frame(
    tracking_df: pd.DataFrame,
    pids,
    frame_id: int,
    w_dir: float = 0.5,
    w_spd: float = 0.3,
    w_dacc: float = 0.2,
    vmin: float = 1.2,
    min_dir_active: int = 2
) -> dict:
    frame_id = _resolve_frame_id(tracking_df, frame_id)
    row = tracking_df.set_index("frame_id").loc[frame_id]
    comps = group_sync_metric(
        row, pids, w_dir=w_dir, w_spd=w_spd, w_dacc=w_dacc, vmin=vmin,
        min_dir_active=min_dir_active
    )
    comps["frame_id"] = int(frame_id)
    return comps


def group_sync_over_time(
    tracking_df: pd.DataFrame,
    pids,
    frame_ids: list[int] | None = None,
    start_time=None,
    end_time=None,
    w_dir: float = 0.5,
    w_spd: float = 0.3,
    w_dacc: float = 0.2,
    vmin: float = 1.2,
    min_dir_active: int = 2
) -> pd.DataFrame:
    frame_ids = _frame_list_from_inputs(tracking_df, frame_ids, start_time, end_time)
    frames_idx = tracking_df.set_index("frame_id")
    out = []
    for fid in frame_ids:
        if fid not in frames_idx.index:
            continue
        row = frames_idx.loc[fid]
        comps = group_sync_metric(
            row, pids, w_dir=w_dir, w_spd=w_spd, w_dacc=w_dacc, vmin=vmin,
            min_dir_active=min_dir_active
        )
        comps["frame_id"] = int(fid)
        out.append(comps)
    return pd.DataFrame(out)


def plot_triangle_synchrony_on_pitch_mpl(
    tracking_team_df: pd.DataFrame,
    sync_df: pd.DataFrame,
    frame_id: int,
    pitch_length: float = 105,
    pitch_width: float = 68,
    cmap: str = "RdYlGn",
    annotate_players: bool = True,
    show_values: bool = True,
    vmin: float = 0.0,
    vmax: float = 1.0,
    player_size: float = 220,
    opponent_df: pd.DataFrame | None = None,
    show_opponents: bool = True,
    show_ball: bool = True,
    show_ball_vector: bool = True,
    ball_size: float = 240,
    show_vectors: bool = True,
    speed_min_arrow: float = 0.8,
    arrow_speed_scale: float = 1.2,
    arrow_scale: float = 6.0,
    arrow_width: float = 0.003,
    value_offset_if_small: float = 1.0,
    small_area_thresh: float = 12.0
):
    frame_id = _resolve_frame_id(tracking_team_df, frame_id, required_frames=set(sync_df["frame_id"].astype(int)))
    frame = tracking_team_df.loc[tracking_team_df["frame_id"] == frame_id]
    r = frame.iloc[0]

    bases = _player_bases_only_numeric(tracking_team_df)
    player_pos_c, player_vel = {}, {}
    for b in bases:
        x = r.get(f"{b}_x", np.nan); y = r.get(f"{b}_y", np.nan)
        if np.isfinite(x) and np.isfinite(y):
            player_pos_c[int(b)] = (float(x), float(y))
            vx = r.get(f"{b}_vx", np.nan); vy = r.get(f"{b}_vy", np.nan)
            player_vel[int(b)] = (float(vx), float(vy)) if np.isfinite(vx) and np.isfinite(vy) else (np.nan, np.nan)

    if not player_pos_c:
        raise ValueError("No player positions for this frame.")

    tris = sync_df[sync_df["frame_id"] == frame_id]
    if tris.empty:
        raise ValueError(f"No synchrony data for frame_id={frame_id}")

    polygons_c, colors, centers_c, areas = [], [], [], []
    for _, t in tris.iterrows():
        pts = []
        for pid in (t["pid1"], t["pid2"], t["pid3"]):
            if pid in player_pos_c:
                pts.append(player_pos_c[pid])
        if len(pts) == 3:
            P = np.array(pts, dtype=float)
            area = 0.5 * abs(
                P[0, 0] * (P[1, 1] - P[2, 1]) +
                P[1, 0] * (P[2, 1] - P[0, 1]) +
                P[2, 0] * (P[0, 1] - P[1, 1])
            )
            polygons_c.append(P)
            colors.append(float(t["Sync"]))
            centers_c.append(P.mean(axis=0))
            areas.append(area)

    if not polygons_c:
        raise ValueError("No complete triangles to draw.")

    polygons = [_to_mpl(p, pitch_length, pitch_width) for p in polygons_c]
    centers = _to_mpl(np.vstack(centers_c), pitch_length, pitch_width)
    players_xy = _to_mpl(np.array(list(player_pos_c.values())), pitch_length, pitch_width)
    player_ids = list(player_pos_c.keys())

    V = np.array([player_vel[pid] for pid in player_ids])
    speeds = np.hypot(V[:, 0], V[:, 1])

    pitch = Pitch(pitch_length=pitch_length, pitch_width=pitch_width, line_zorder=1)
    fig, ax = pitch.draw(figsize=(12, 7))
    ax.set_title(f"Triangle synchrony - frame {frame_id}", fontsize=14)

    if opponent_df is not None and show_opponents:
        opp_row = opponent_df.loc[opponent_df["frame_id"] == frame_id]
        if not opp_row.empty:
            opp_row = opp_row.iloc[0]
            opp_bases = _player_bases_only_numeric(opponent_df)
            opp_pts, _ = _coords_for_frame(opp_row, opp_bases)
            if opp_pts.size:
                opp_plot = _to_mpl(opp_pts, pitch_length, pitch_width)
                pitch.scatter(opp_plot[:, 0], opp_plot[:, 1], ax=ax, s=160, marker="o",
                              facecolor="#d62728", edgecolor="black", zorder=1.5, label="Opponent")

    pc = PolyCollection(polygons, array=np.array(colors), cmap=cmap,
                        edgecolor="black", lw=1.0, alpha=0.65, zorder=2)
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)

    cbar = plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Synchrony (0-1)")

    pitch.scatter(players_xy[:, 0], players_xy[:, 1], ax=ax, s=player_size,
                  marker="o", facecolor="#1f77b4", edgecolor="black", zorder=3)

    if annotate_players:
        for (x, y), pid in zip(players_xy, player_ids):
            ax.text(x, y + 0.9, str(pid), ha="center", va="bottom", fontsize=9, color="black", zorder=4)

    if show_vectors:
        mask = np.isfinite(speeds) & (speeds >= speed_min_arrow)
        if mask.any():
            X = players_xy[mask, 0]; Y = players_xy[mask, 1]
            Vsel = V[mask]
            spd = speeds[mask]
            denom = np.maximum(spd, 1e-9)
            U = (Vsel[:, 0] / denom) * (spd * arrow_speed_scale)
            W = (Vsel[:, 1] / denom) * (spd * arrow_speed_scale)
            ax.quiver(X, Y, U, W, angles="xy", scale_units="xy",
                      scale=1.0, width=arrow_width, color="black", zorder=4)

    if show_ball and {"ball_x", "ball_y"}.issubset(tracking_team_df.columns):
        bx, by = float(r["ball_x"]), float(r["ball_y"])
        ball_xy = _to_mpl(np.array([[bx, by]], dtype=float), pitch_length, pitch_width)[0]
        ax.scatter(ball_xy[0], ball_xy[1], s=ball_size, c="#fffb00", edgecolor="black",
                   marker="o", zorder=5)
        if show_ball_vector and {"ball_vx", "ball_vy"}.issubset(tracking_team_df.columns):
            bvx = r.get("ball_vx", np.nan); bvy = r.get("ball_vy", np.nan)
            if np.isfinite(bvx) and np.isfinite(bvy) and np.hypot(bvx, bvy) > 0.1:
                ax.quiver(ball_xy[0], ball_xy[1], bvx, bvy, angles="xy", scale_units="xy",
                          scale=arrow_scale, width=arrow_width, color="#ff7f0e", zorder=5)

    if show_values:
        for (cx, cy), val, area in zip(centers, colors, areas):
            ty = cy + (value_offset_if_small if area < small_area_thresh else 0.0)
            ax.text(cx, ty, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color="black", weight="bold", zorder=6)

    return fig, ax


def plot_group_sync_on_pitch(
    tracking_team_df: pd.DataFrame,
    pids: list[int],
    frame_id: int,
    pitch_length: float = 105,
    pitch_width: float = 68,
    show_all_players: bool = True,
    opponent_df: pd.DataFrame | None = None,
    show_opponents: bool = True,
    annotate_players: bool = True,
    show_sync_text: bool = True,
    show_ball: bool = True,
    show_ball_vector: bool = True,
    ball_size: float = 240,
    show_vectors: bool = True,
    speed_min_arrow: float = 0.8,
    arrow_speed_scale: float = 1.2,
    arrow_width: float = 0.003,
    player_size: float = 220,
    group_alpha: float = 0.4,
    group_edgecolor: str = "black",
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0
):
    frame_id = _resolve_frame_id(tracking_team_df, frame_id)
    r = tracking_team_df.loc[tracking_team_df["frame_id"] == frame_id].iloc[0]

    player_pos_c, player_vel = {}, {}
    for pid in pids:
        x = r.get(f"{pid}_x", np.nan); y = r.get(f"{pid}_y", np.nan)
        if np.isfinite(x) and np.isfinite(y):
            player_pos_c[int(pid)] = (float(x), float(y))
            vx = r.get(f"{pid}_vx", np.nan); vy = r.get(f"{pid}_vy", np.nan)
            player_vel[int(pid)] = (float(vx), float(vy)) if np.isfinite(vx) and np.isfinite(vy) else (np.nan, np.nan)

    if not player_pos_c:
        raise ValueError("No positions for this group in this frame.")

    players_xy = _to_mpl(np.array(list(player_pos_c.values())), pitch_length, pitch_width)
    player_ids = list(player_pos_c.keys())
    V = np.array([player_vel[pid] for pid in player_ids])
    speeds = np.hypot(V[:, 0], V[:, 1])

    pitch = Pitch(pitch_length=pitch_length, pitch_width=pitch_width, line_zorder=1)
    fig, ax = pitch.draw(figsize=(12, 7))
    ax.set_title(f"Group sync - frame {frame_id}", fontsize=14)

    all_xy = None
    all_ids = None
    if show_all_players:
        bases = _player_bases_only_numeric(tracking_team_df)
        all_pos = []
        all_ids_list = []
        for b in bases:
            x = r.get(f"{b}_x", np.nan); y = r.get(f"{b}_y", np.nan)
            if np.isfinite(x) and np.isfinite(y):
                all_pos.append((float(x), float(y)))
                all_ids_list.append(int(b))
        if all_pos:
            all_xy = _to_mpl(np.array(all_pos), pitch_length, pitch_width)
            all_ids = all_ids_list
            pitch.scatter(all_xy[:, 0], all_xy[:, 1], ax=ax, s=140,
                          marker="o", facecolor="#1f77b4", edgecolor="black", zorder=2)

    if opponent_df is not None and show_opponents:
        opp_row = opponent_df.loc[opponent_df["frame_id"] == frame_id]
        if not opp_row.empty:
            opp_row = opp_row.iloc[0]
            opp_bases = _player_bases_only_numeric(opponent_df)
            opp_pos = []
            for b in opp_bases:
                x = opp_row.get(f"{b}_x", np.nan); y = opp_row.get(f"{b}_y", np.nan)
                if np.isfinite(x) and np.isfinite(y):
                    opp_pos.append((float(x), float(y)))
            if opp_pos:
                opp_xy = _to_mpl(np.array(opp_pos), pitch_length, pitch_width)
                pitch.scatter(opp_xy[:, 0], opp_xy[:, 1], ax=ax, s=140,
                              marker="o", facecolor="#d62728", edgecolor="black", zorder=2)

    pitch.scatter(players_xy[:, 0], players_xy[:, 1], ax=ax, s=player_size,
                  marker="o", facecolor="#1f77b4", edgecolor="black", zorder=3)

    if annotate_players:
        if show_all_players and all_xy is not None and all_ids is not None:
            for (x, y), pid in zip(all_xy, all_ids):
                ax.text(x, y + 0.9, str(pid), ha="center", va="bottom", fontsize=8, color="black", zorder=3)
        else:
            for (x, y), pid in zip(players_xy, player_ids):
                ax.text(x, y + 0.9, str(pid), ha="center", va="bottom", fontsize=9, color="black", zorder=4)

    if show_vectors:
        mask = np.isfinite(speeds) & (speeds >= speed_min_arrow)
        if mask.any():
            X = players_xy[mask, 0]; Y = players_xy[mask, 1]
            Vsel = V[mask]
            spd = speeds[mask]
            denom = np.maximum(spd, 1e-9)
            U = (Vsel[:, 0] / denom) * (spd * arrow_speed_scale)
            W = (Vsel[:, 1] / denom) * (spd * arrow_speed_scale)
            ax.quiver(X, Y, U, W, angles="xy", scale_units="xy",
                      scale=1.0, width=arrow_width, color="black", zorder=4)

    if show_ball and {"ball_x", "ball_y"}.issubset(tracking_team_df.columns):
        bx, by = float(r["ball_x"]), float(r["ball_y"])
        ball_xy = _to_mpl(np.array([[bx, by]], dtype=float), pitch_length, pitch_width)[0]
        ax.scatter(ball_xy[0], ball_xy[1], s=ball_size, c="#fffb00", edgecolor="black",
                   marker="o", zorder=5, label="Ball")
        if show_ball_vector and {"ball_vx", "ball_vy"}.issubset(tracking_team_df.columns):
            bvx = r.get("ball_vx", np.nan); bvy = r.get("ball_vy", np.nan)
            if np.isfinite(bvx) and np.isfinite(bvy) and np.hypot(bvx, bvy) > 0.1:
                ax.quiver(ball_xy[0], ball_xy[1], bvx, bvy, angles="xy", scale_units="xy",
                          scale=6.0, width=arrow_width, color="#ff7f0e", zorder=5)

    comps = group_sync_for_frame(tracking_team_df, pids, frame_id)
    center = players_xy.mean(axis=0)

    if players_xy.shape[0] >= 3:
        angles = np.arctan2(players_xy[:, 1] - center[1], players_xy[:, 0] - center[0])
        order = np.argsort(angles)
        poly = players_xy[order]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        cmap_obj = plt.get_cmap(cmap)
        if np.isfinite(comps.get("Sync", np.nan)):
            face = cmap_obj(norm(float(comps["Sync"])))
        else:
            face = "#cccccc"
        pc = PolyCollection([poly], facecolor=[face], edgecolor=group_edgecolor,
                            alpha=group_alpha, zorder=2.5)
        ax.add_collection(pc)

    if show_sync_text:
        ax.text(center[0], center[1], f"{comps['Sync']:.2f}",
                ha="center", va="center", fontsize=11, weight="bold", zorder=6)

    return fig, ax


def animate_triangle_synchrony_on_pitch_mpl(
    tracking_team_df: pd.DataFrame,
    sync_df: pd.DataFrame,
    frame_ids: list[int] | None = None,
    start_time=None,
    end_time=None,
    interval: int = 200,
    pitch_length: float = 105,
    pitch_width: float = 68,
    cmap: str = "RdYlGn",
    annotate_players: bool = True,
    show_values: bool = True,
    vmin: float = 0.0,
    vmax: float = 1.0,
    player_size: float = 220,
    opponent_df: pd.DataFrame | None = None,
    show_opponents: bool = True,
    show_ball: bool = True,
    show_ball_vector: bool = True,
    ball_size: float = 240,
    show_vectors: bool = True,
    speed_min_arrow: float = 0.8,
    arrow_speed_scale: float = 1.2,
    arrow_width: float = 0.003
):
    frames = _frame_list_from_inputs(tracking_team_df, frame_ids, start_time, end_time)
    avail = set(sync_df["frame_id"].astype(int).tolist())
    track_frames = set(tracking_team_df["frame_id"].astype(int).tolist())
    frames = [f for f in frames if f in avail and f in track_frames]
    if not frames:
        raise ValueError("No available frame_id for animation.")

    pitch = Pitch(pitch_length=pitch_length, pitch_width=pitch_width, line_zorder=1)
    fig, ax = pitch.draw(figsize=(12, 7))

    def _draw_frame(fid):
        ax.clear()
        pitch.draw(ax=ax)
        ax.set_title(f"Triangle synchrony - frame {fid}", fontsize=14)

        row = tracking_team_df.loc[tracking_team_df["frame_id"] == fid].iloc[0]
        bases = _player_bases_only_numeric(tracking_team_df)
        player_pos_c, player_vel = {}, {}
        for b in bases:
            x = row.get(f"{b}_x", np.nan); y = row.get(f"{b}_y", np.nan)
            if np.isfinite(x) and np.isfinite(y):
                player_pos_c[int(b)] = (float(x), float(y))
                vx = row.get(f"{b}_vx", np.nan); vy = row.get(f"{b}_vy", np.nan)
                player_vel[int(b)] = (float(vx), float(vy)) if np.isfinite(vx) and np.isfinite(vy) else (np.nan, np.nan)

        if opponent_df is not None and show_opponents:
            opp_row = opponent_df.loc[opponent_df["frame_id"] == fid]
            if not opp_row.empty:
                opp_row = opp_row.iloc[0]
                opp_bases = _player_bases_only_numeric(opponent_df)
                opp_pts, _ = _coords_for_frame(opp_row, opp_bases)
                if opp_pts.size:
                    opp_plot = _to_mpl(opp_pts, pitch_length, pitch_width)
                    pitch.scatter(opp_plot[:, 0], opp_plot[:, 1], ax=ax, s=160, marker="o",
                                  facecolor="#d62728", edgecolor="black", zorder=1.5)

        if not player_pos_c:
            return []

        players_xy = _to_mpl(np.array(list(player_pos_c.values())), pitch_length, pitch_width)
        player_ids = list(player_pos_c.keys())
        V = np.array([player_vel[pid] for pid in player_ids])
        speeds = np.hypot(V[:, 0], V[:, 1])

        tris = sync_df[sync_df["frame_id"] == fid]
        polygons, colors, centers, areas = [], [], [], []
        for _, t in tris.iterrows():
            pts = []
            for pid in (t["pid1"], t["pid2"], t["pid3"]):
                if pid in player_pos_c:
                    pts.append(player_pos_c[pid])
            if len(pts) == 3:
                P = np.array(pts, dtype=float)
                area = 0.5 * abs(
                    P[0, 0] * (P[1, 1] - P[2, 1]) +
                    P[1, 0] * (P[2, 1] - P[0, 1]) +
                    P[2, 0] * (P[0, 1] - P[1, 1])
                )
                polygons.append(_to_mpl(P, pitch_length, pitch_width))
                colors.append(float(t["Sync"]))
                centers.append(_to_mpl(P.mean(axis=0, keepdims=True), pitch_length, pitch_width)[0])
                areas.append(area)

        if polygons:
            pc = PolyCollection(polygons, array=np.array(colors), cmap=cmap,
                                edgecolor="black", lw=1.0, alpha=0.65, zorder=2)
            pc.set_clim(vmin, vmax)
            ax.add_collection(pc)

        pitch.scatter(players_xy[:, 0], players_xy[:, 1], ax=ax, s=player_size,
                      marker="o", facecolor="#1f77b4", edgecolor="black", zorder=3)

        if annotate_players:
            for (x, y), pid in zip(players_xy, player_ids):
                ax.text(x, y + 0.9, str(pid), ha="center", va="bottom", fontsize=9, color="black", zorder=4)

        if show_vectors:
            mask = np.isfinite(speeds) & (speeds >= speed_min_arrow)
            if mask.any():
                X = players_xy[mask, 0]; Y = players_xy[mask, 1]
                Vsel = V[mask]
                spd = speeds[mask]
                denom = np.maximum(spd, 1e-9)
                U = (Vsel[:, 0] / denom) * (spd * arrow_speed_scale)
                W = (Vsel[:, 1] / denom) * (spd * arrow_speed_scale)
                ax.quiver(X, Y, U, W, angles="xy", scale_units="xy",
                          scale=1.0, width=arrow_width, color="black", zorder=4)

        if show_ball and {"ball_x", "ball_y"}.issubset(tracking_team_df.columns):
            bx, by = float(row["ball_x"]), float(row["ball_y"])
            ball_xy = _to_mpl(np.array([[bx, by]], dtype=float), pitch_length, pitch_width)[0]
            ax.scatter(ball_xy[0], ball_xy[1], s=ball_size, c="#fffb00", edgecolor="black",
                       marker="o", zorder=5)
            if show_ball_vector and {"ball_vx", "ball_vy"}.issubset(tracking_team_df.columns):
                bvx = row.get("ball_vx", np.nan); bvy = row.get("ball_vy", np.nan)
                if np.isfinite(bvx) and np.isfinite(bvy) and np.hypot(bvx, bvy) > 0.1:
                    ax.quiver(ball_xy[0], ball_xy[1], bvx, bvy, angles="xy", scale_units="xy",
                              scale=6.0, width=arrow_width, color="#ff7f0e", zorder=5)

        if show_values and centers:
            for (cx, cy), val, area in zip(centers, colors, areas):
                ty = cy + (1.0 if area < 12.0 else 0.0)
                ax.text(cx, ty, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="black", weight="bold", zorder=6)

        return []

    anim = mpl_animation.FuncAnimation(fig, lambda i: _draw_frame(frames[i]),
                                        frames=len(frames), interval=interval, blit=False)
    return fig, anim


def animate_group_sync_on_pitch(
    tracking_team_df: pd.DataFrame,
    pids: list[int],
    frame_ids: list[int] | None = None,
    start_time=None,
    end_time=None,
    interval: int = 200,
    pitch_length: float = 105,
    pitch_width: float = 68,
    show_all_players: bool = True,
    opponent_df: pd.DataFrame | None = None,
    show_opponents: bool = True,
    annotate_players: bool = True,
    show_sync_text: bool = True,
    show_ball: bool = True,
    show_ball_vector: bool = True,
    ball_size: float = 240,
    show_vectors: bool = True,
    speed_min_arrow: float = 0.8,
    arrow_speed_scale: float = 1.2,
    arrow_width: float = 0.003,
    player_size: float = 220,
    group_alpha: float = 0.4,
    group_edgecolor: str = "black",
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0
):
    frames = _frame_list_from_inputs(tracking_team_df, frame_ids, start_time, end_time)
    track_frames = set(tracking_team_df["frame_id"].astype(int).tolist())
    frames = [f for f in frames if f in track_frames]
    if not frames:
        raise ValueError("No available frame_id for animation.")

    pitch = Pitch(pitch_length=pitch_length, pitch_width=pitch_width, line_zorder=1)
    fig, ax = pitch.draw(figsize=(12, 7))

    def _draw_frame(fid):
        ax.clear()
        pitch.draw(ax=ax)
        ax.set_title(f"Group sync - frame {fid}", fontsize=14)

        row = tracking_team_df.loc[tracking_team_df["frame_id"] == fid].iloc[0]

        all_xy = None
        all_ids = None
        if show_all_players:
            bases = _player_bases_only_numeric(tracking_team_df)
            all_pos = []
            all_ids_list = []
            for b in bases:
                x = row.get(f"{b}_x", np.nan); y = row.get(f"{b}_y", np.nan)
                if np.isfinite(x) and np.isfinite(y):
                    all_pos.append((float(x), float(y)))
                    all_ids_list.append(int(b))
            if all_pos:
                all_xy = _to_mpl(np.array(all_pos), pitch_length, pitch_width)
                all_ids = all_ids_list
                pitch.scatter(all_xy[:, 0], all_xy[:, 1], ax=ax, s=140,
                              marker="o", facecolor="#1f77b4", edgecolor="black", zorder=2)

        if opponent_df is not None and show_opponents:
            opp_row = opponent_df.loc[opponent_df["frame_id"] == fid]
            if not opp_row.empty:
                opp_row = opp_row.iloc[0]
                opp_bases = _player_bases_only_numeric(opponent_df)
                opp_pos = []
                for b in opp_bases:
                    x = opp_row.get(f"{b}_x", np.nan); y = opp_row.get(f"{b}_y", np.nan)
                    if np.isfinite(x) and np.isfinite(y):
                        opp_pos.append((float(x), float(y)))
                if opp_pos:
                    opp_xy = _to_mpl(np.array(opp_pos), pitch_length, pitch_width)
                    pitch.scatter(opp_xy[:, 0], opp_xy[:, 1], ax=ax, s=140,
                                  marker="o", facecolor="#d62728", edgecolor="black", zorder=2)

        player_pos_c, player_vel = {}, {}
        for pid in pids:
            x = row.get(f"{pid}_x", np.nan); y = row.get(f"{pid}_y", np.nan)
            if np.isfinite(x) and np.isfinite(y):
                player_pos_c[int(pid)] = (float(x), float(y))
                vx = row.get(f"{pid}_vx", np.nan); vy = row.get(f"{pid}_vy", np.nan)
                player_vel[int(pid)] = (float(vx), float(vy)) if np.isfinite(vx) and np.isfinite(vy) else (np.nan, np.nan)

        if not player_pos_c:
            return []

        players_xy = _to_mpl(np.array(list(player_pos_c.values())), pitch_length, pitch_width)
        player_ids = list(player_pos_c.keys())
        V = np.array([player_vel[pid] for pid in player_ids])
        speeds = np.hypot(V[:, 0], V[:, 1])

        pitch.scatter(players_xy[:, 0], players_xy[:, 1], ax=ax, s=player_size,
                      marker="o", facecolor="#1f77b4", edgecolor="black", zorder=3)

        if annotate_players:
            if show_all_players and all_xy is not None and all_ids is not None:
                for (x, y), pid in zip(all_xy, all_ids):
                    ax.text(x, y + 0.9, str(pid), ha="center", va="bottom", fontsize=8, color="black", zorder=3)
            else:
                for (x, y), pid in zip(players_xy, player_ids):
                    ax.text(x, y + 0.9, str(pid), ha="center", va="bottom", fontsize=9, color="black", zorder=4)

        if show_vectors:
            mask = np.isfinite(speeds) & (speeds >= speed_min_arrow)
            if mask.any():
                X = players_xy[mask, 0]; Y = players_xy[mask, 1]
                Vsel = V[mask]
                spd = speeds[mask]
                denom = np.maximum(spd, 1e-9)
                U = (Vsel[:, 0] / denom) * (spd * arrow_speed_scale)
                W = (Vsel[:, 1] / denom) * (spd * arrow_speed_scale)
                ax.quiver(X, Y, U, W, angles="xy", scale_units="xy",
                          scale=1.0, width=arrow_width, color="black", zorder=4)

        if show_ball and {"ball_x", "ball_y"}.issubset(tracking_team_df.columns):
            bx, by = float(row["ball_x"]), float(row["ball_y"])
            ball_xy = _to_mpl(np.array([[bx, by]], dtype=float), pitch_length, pitch_width)[0]
            ax.scatter(ball_xy[0], ball_xy[1], s=ball_size, c="#fffb00", edgecolor="black",
                       marker="o", zorder=5)
            if show_ball_vector and {"ball_vx", "ball_vy"}.issubset(tracking_team_df.columns):
                bvx = row.get("ball_vx", np.nan); bvy = row.get("ball_vy", np.nan)
                if np.isfinite(bvx) and np.isfinite(bvy) and np.hypot(bvx, bvy) > 0.1:
                    ax.quiver(ball_xy[0], ball_xy[1], bvx, bvy, angles="xy", scale_units="xy",
                              scale=6.0, width=arrow_width, color="#ff7f0e", zorder=5)

        comps = group_sync_for_frame(tracking_team_df, pids, fid)
        center = players_xy.mean(axis=0)

        if players_xy.shape[0] >= 3:
            angles = np.arctan2(players_xy[:, 1] - center[1], players_xy[:, 0] - center[0])
            order = np.argsort(angles)
            poly = players_xy[order]
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
            cmap_obj = plt.get_cmap(cmap)
            if np.isfinite(comps.get("Sync", np.nan)):
                face = cmap_obj(norm(float(comps["Sync"])))
            else:
                face = "#cccccc"
            pc = PolyCollection([poly], facecolor=[face], edgecolor=group_edgecolor,
                                alpha=group_alpha, zorder=2.5)
            ax.add_collection(pc)

        if show_sync_text:
            ax.text(center[0], center[1], f"{comps['Sync']:.2f}",
                    ha="center", va="center", fontsize=11, weight="bold", zorder=6)

        return []

    anim = mpl_animation.FuncAnimation(fig, lambda i: _draw_frame(frames[i]),
                                        frames=len(frames), interval=interval, blit=False)
    return fig, anim


class synchTracking:
    @staticmethod
    def build_delaunay_per_frame(tracking_team_df, min_points: int = 3):
        return build_delaunay_per_frame(tracking_team_df, min_points=min_points)

    @staticmethod
    def compute_player_kinematics(
        tracking_df, fps: float, smooth: str = "sg", window: int = 7, poly: int = 2,
        interpolate_gaps: bool = True, max_gap: int = 5,
        direction_min_speed: float = 0.3, compute_ball: bool = True
    ):
        return compute_player_kinematics(
            tracking_df, fps, smooth=smooth, window=window, poly=poly,
            interpolate_gaps=interpolate_gaps, max_gap=max_gap,
            direction_min_speed=direction_min_speed, compute_ball=compute_ball
        )

    @staticmethod
    def compute_triangle_synchrony(
        df, triangles_by_frame, w_dir: float = 0.5, w_spd: float = 0.3,
        w_dacc: float = 0.2, vmin: float = 1.2
    ):
        return compute_triangle_synchrony(
            df, triangles_by_frame, w_dir=w_dir, w_spd=w_spd, w_dacc=w_dacc, vmin=vmin
        )

    @staticmethod
    def group_sync_for_frame(
        tracking_df, pids, frame_id: int, w_dir: float = 0.5, w_spd: float = 0.3,
        w_dacc: float = 0.2, vmin: float = 1.2, min_dir_active: int = 2
    ):
        return group_sync_for_frame(
            tracking_df, pids, frame_id, w_dir=w_dir, w_spd=w_spd, w_dacc=w_dacc,
            vmin=vmin, min_dir_active=min_dir_active
        )

    @staticmethod
    def group_sync_over_time(
        tracking_df, pids, frame_ids: list[int] | None = None, start_time=None, end_time=None,
        w_dir: float = 0.5, w_spd: float = 0.3, w_dacc: float = 0.2, vmin: float = 1.2,
        min_dir_active: int = 2
    ):
        return group_sync_over_time(
            tracking_df, pids, frame_ids=frame_ids, start_time=start_time, end_time=end_time,
            w_dir=w_dir, w_spd=w_spd, w_dacc=w_dacc, vmin=vmin, min_dir_active=min_dir_active
        )

    @staticmethod
    def compute_sync_for_time_range(
        tracking_df, fps: float, start_time=None, end_time=None, buffer_frames: int = 5,
        smooth: str = "sg", window: int = 7, poly: int = 2,
        direction_min_speed: float = 0.3, compute_ball: bool = True, min_points: int = 3,
        w_dir: float = 0.5, w_spd: float = 0.3, w_dacc: float = 0.2, vmin: float = 1.2
    ):
        return compute_sync_for_time_range(
            tracking_df, fps, start_time=start_time, end_time=end_time, buffer_frames=buffer_frames,
            smooth=smooth, window=window, poly=poly, direction_min_speed=direction_min_speed,
            compute_ball=compute_ball, min_points=min_points, w_dir=w_dir, w_spd=w_spd,
            w_dacc=w_dacc, vmin=vmin
        )

    @staticmethod
    def frame_ids_from_time_range(tracking_df, start_time=None, end_time=None):
        return frame_ids_from_time_range(tracking_df, start_time=start_time, end_time=end_time)

    @staticmethod
    def filter_triangles_and_sync_by_time(tracking_df, triangles_by_frame, sync_df, start_time=None, end_time=None):
        return filter_triangles_and_sync_by_time(
            tracking_df, triangles_by_frame, sync_df, start_time=start_time, end_time=end_time
        )

    @staticmethod
    def team_sync_for_frame(sync_df, frame_id: int, metric: str = "Sync", agg: str = "mean", min_triangles: int = 1):
        return team_sync_for_frame(sync_df, frame_id, metric=metric, agg=agg, min_triangles=min_triangles)

    @staticmethod
    def team_sync_over_time(sync_df, metric: str = "Sync", agg: str = "mean", min_triangles: int = 1):
        return team_sync_over_time(sync_df, metric=metric, agg=agg, min_triangles=min_triangles)

    @staticmethod
    def plot_team_triangulation_on_frame(
        tracking_team_df, triangles_by_frame, frame_id: int,
        pitch_length: float = 104, pitch_width: float = 68,
        annotate_ids: bool = True, opponent_df: pd.DataFrame | None = None,
        show_opponents: bool = True, figsize=(10, 6)
    ):
        return plot_team_triangulation_on_frame(
            tracking_team_df, triangles_by_frame, frame_id,
            pitch_length=pitch_length, pitch_width=pitch_width,
            annotate_ids=annotate_ids, opponent_df=opponent_df,
            show_opponents=show_opponents, figsize=figsize
        )

    @staticmethod
    def plot_group_sync_on_pitch(
        tracking_team_df, pids, frame_id: int,
        pitch_length: float = 105, pitch_width: float = 68,
        show_all_players: bool = True, opponent_df: pd.DataFrame | None = None, show_opponents: bool = True,
        annotate_players: bool = True, show_sync_text: bool = True,
        show_ball: bool = True, show_ball_vector: bool = True, ball_size: float = 240,
        show_vectors: bool = True, speed_min_arrow: float = 0.8,
        arrow_speed_scale: float = 1.2, arrow_width: float = 0.003,
        player_size: float = 220, group_alpha: float = 0.4,
        group_edgecolor: str = "black", cmap: str = "RdYlGn",
        vmin: float = 0.0, vmax: float = 1.0
    ):
        return plot_group_sync_on_pitch(
            tracking_team_df, pids, frame_id,
            pitch_length=pitch_length, pitch_width=pitch_width,
            show_all_players=show_all_players, opponent_df=opponent_df, show_opponents=show_opponents,
            annotate_players=annotate_players, show_sync_text=show_sync_text,
            show_ball=show_ball, show_ball_vector=show_ball_vector, ball_size=ball_size,
            show_vectors=show_vectors, speed_min_arrow=speed_min_arrow,
            arrow_speed_scale=arrow_speed_scale, arrow_width=arrow_width,
            player_size=player_size, group_alpha=group_alpha,
            group_edgecolor=group_edgecolor, cmap=cmap, vmin=vmin, vmax=vmax
        )

    @staticmethod
    def animate_group_sync_on_pitch(
        tracking_team_df, pids, frame_ids: list[int] | None = None, start_time=None, end_time=None,
        interval: int = 200, pitch_length: float = 105, pitch_width: float = 68,
        show_all_players: bool = True, opponent_df: pd.DataFrame | None = None, show_opponents: bool = True,
        annotate_players: bool = True, show_sync_text: bool = True, show_ball: bool = True,
        show_ball_vector: bool = True, ball_size: float = 240, show_vectors: bool = True,
        speed_min_arrow: float = 0.8, arrow_speed_scale: float = 1.2, arrow_width: float = 0.003,
        player_size: float = 220, group_alpha: float = 0.4, group_edgecolor: str = "black",
        cmap: str = "RdYlGn", vmin: float = 0.0, vmax: float = 1.0
    ):
        return animate_group_sync_on_pitch(
            tracking_team_df, pids, frame_ids=frame_ids, start_time=start_time, end_time=end_time,
            interval=interval, pitch_length=pitch_length, pitch_width=pitch_width,
            show_all_players=show_all_players, opponent_df=opponent_df, show_opponents=show_opponents,
            annotate_players=annotate_players, show_sync_text=show_sync_text, show_ball=show_ball,
            show_ball_vector=show_ball_vector, ball_size=ball_size, show_vectors=show_vectors,
            speed_min_arrow=speed_min_arrow, arrow_speed_scale=arrow_speed_scale, arrow_width=arrow_width,
            player_size=player_size, group_alpha=group_alpha, group_edgecolor=group_edgecolor,
            cmap=cmap, vmin=vmin, vmax=vmax
        )

    @staticmethod
    def plot_triangle_synchrony_on_pitch_mpl(
        tracking_team_df, sync_df, frame_id: int,
        pitch_length: float = 105, pitch_width: float = 68, cmap: str = "RdYlGn",
        annotate_players: bool = True, show_values: bool = True,
        vmin: float = 0.0, vmax: float = 1.0, player_size: float = 220,
        opponent_df: pd.DataFrame | None = None, show_opponents: bool = True,
        show_ball: bool = True, show_ball_vector: bool = True, ball_size: float = 240,
        show_vectors: bool = True, speed_min_arrow: float = 0.8,
        arrow_speed_scale: float = 1.2, arrow_scale: float = 6.0,
        arrow_width: float = 0.003, value_offset_if_small: float = 1.0,
        small_area_thresh: float = 12.0
    ):
        return plot_triangle_synchrony_on_pitch_mpl(
            tracking_team_df, sync_df, frame_id,
            pitch_length=pitch_length, pitch_width=pitch_width, cmap=cmap,
            annotate_players=annotate_players, show_values=show_values,
            vmin=vmin, vmax=vmax, player_size=player_size,
            opponent_df=opponent_df, show_opponents=show_opponents,
            show_ball=show_ball, show_ball_vector=show_ball_vector, ball_size=ball_size,
            show_vectors=show_vectors, speed_min_arrow=speed_min_arrow,
            arrow_speed_scale=arrow_speed_scale, arrow_scale=arrow_scale,
            arrow_width=arrow_width, value_offset_if_small=value_offset_if_small,
            small_area_thresh=small_area_thresh
        )

    @staticmethod
    def animate_triangle_synchrony_on_pitch_mpl(
        tracking_team_df, sync_df, frame_ids: list[int] | None = None, start_time=None, end_time=None,
        interval: int = 200, pitch_length: float = 105, pitch_width: float = 68, cmap: str = "RdYlGn",
        annotate_players: bool = True, show_values: bool = True, vmin: float = 0.0, vmax: float = 1.0,
        player_size: float = 220, opponent_df: pd.DataFrame | None = None, show_opponents: bool = True,
        show_ball: bool = True, show_ball_vector: bool = True, ball_size: float = 240,
        show_vectors: bool = True, speed_min_arrow: float = 0.8, arrow_speed_scale: float = 1.2,
        arrow_width: float = 0.003
    ):
        return animate_triangle_synchrony_on_pitch_mpl(
            tracking_team_df, sync_df, frame_ids=frame_ids, start_time=start_time, end_time=end_time,
            interval=interval, pitch_length=pitch_length, pitch_width=pitch_width, cmap=cmap,
            annotate_players=annotate_players, show_values=show_values, vmin=vmin, vmax=vmax,
            player_size=player_size, opponent_df=opponent_df, show_opponents=show_opponents,
            show_ball=show_ball, show_ball_vector=show_ball_vector, ball_size=ball_size,
            show_vectors=show_vectors, speed_min_arrow=speed_min_arrow,
            arrow_speed_scale=arrow_speed_scale, arrow_width=arrow_width
        )
