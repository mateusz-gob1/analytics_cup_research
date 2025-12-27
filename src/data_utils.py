import re
import pandas as pd
from kloppy import skillcorner


def _build_players_df(metadata) -> pd.DataFrame:
    rows = []
    for team in metadata.teams:
        team_side = str(team.ground)
        team_id = team.team_id
        team_name = team.name
        for pl in team.players:
            rows.append({
                "player_id": int(pl.player_id) if str(pl.player_id).isdigit() else pl.player_id,
                "player_name": getattr(pl, "name", str(pl.player_id)),
                "team_side": team_side,
                "team_id": team_id,
                "team_name": team_name,
            })
    return pd.DataFrame(rows).set_index("player_id").sort_index()


def load_skillcorner_match(
    match_id: int,
    sample_rate: float = 0.5,
    limit: int | None = None,
    coordinates: str = "skillcorner",
    flatten_tracking: bool = True,
    only_first_half: bool = True,
):
    dataset = skillcorner.load_open_data(
        match_id=match_id,
        sample_rate=sample_rate,
        limit=limit,
        coordinates=coordinates,
    )

    tracking_df = None
    if flatten_tracking:
        ds = dataset.transform(to_orientation="STATIC_HOME_AWAY")
        if only_first_half:
            ds = ds.filter(lambda frame: frame.period.id == 1)
        tracking_df = ds.to_df(engine="pandas")
    events_df = pd.DataFrame()
    phases_df = pd.DataFrame()
    meta = {}

    players_df = _build_players_df(dataset.metadata)
    team_names = {
        "home": next((t.name for t in dataset.metadata.teams if str(t.ground) == "home"), "HOME"),
        "away": next((t.name for t in dataset.metadata.teams if str(t.ground) == "away"), "AWAY"),
    }

    return dataset, tracking_df, events_df, phases_df, meta, players_df, team_names


def _is_player_col(col: str) -> bool:
    return re.fullmatch(r"\d+_(x|y|d|s)", col) is not None


def _player_base_cols(tracking_df: pd.DataFrame) -> set[str]:
    bases_x = {c[:-2] for c in tracking_df.columns if c.endswith("_x")}
    bases_y = {c[:-2] for c in tracking_df.columns if c.endswith("_y")}
    return bases_x & bases_y


def _cols_for_player(pid: str, tracking_df: pd.DataFrame) -> list[str]:
    suffixes = ["x", "y", "d", "s"]
    return [f"{pid}_{suf}" for suf in suffixes if f"{pid}_{suf}" in tracking_df.columns]


def split_tracking_by_team(
    tracking_df: pd.DataFrame,
    players_df: pd.DataFrame,
    keep_general: bool = True,
):
    if keep_general:
        general_cols = [c for c in tracking_df.columns if not _is_player_col(c)]
    else:
        general_cols = []

    bases = _player_base_cols(tracking_df)
    home_ids_all = players_df[players_df["team_side"] == "home"].index.astype(str).tolist()
    away_ids_all = players_df[players_df["team_side"] == "away"].index.astype(str).tolist()

    home_ids_available = [pid for pid in home_ids_all if pid in bases]
    away_ids_available = [pid for pid in away_ids_all if pid in bases]

    home_player_cols = [col for pid in home_ids_available for col in _cols_for_player(pid, tracking_df)]
    away_player_cols = [col for pid in away_ids_available for col in _cols_for_player(pid, tracking_df)]

    tracking_home_df = tracking_df[general_cols + home_player_cols].copy()
    tracking_away_df = tracking_df[general_cols + away_player_cols].copy()

    def _order_cols(general, pid_list):
        ordered = general[:]
        for pid in sorted(pid_list, key=lambda x: int(x) if x.isdigit() else x):
            for suf in ["x", "y", "d", "s"]:
                col = f"{pid}_{suf}"
                if col in tracking_df.columns:
                    ordered.append(col)
        return ordered

    tracking_home_df = tracking_home_df[_order_cols(general_cols, home_ids_available)]
    tracking_away_df = tracking_away_df[_order_cols(general_cols, away_ids_available)]

    return tracking_home_df, tracking_away_df, [int(pid) for pid in home_ids_available], [int(pid) for pid in away_ids_available]

