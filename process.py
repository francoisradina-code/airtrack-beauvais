"""AirTrack Beauvais - nettoyage et calculs sur les données collectées."""
from pathlib import Path

import numpy as np
import pandas as pd

BVA = (49.4544, 2.1128)  # position de l'aéroport LFOB (lat, lon)
M_TO_FT = 3.28084


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def list_days(data_dir: Path):
    """Jours disponibles (dossiers data/AAAA-MM-JJ contenant flights.parquet)."""
    return sorted(p.name for p in data_dir.iterdir()
                  if p.is_dir() and (p / "flights.parquet").exists())


def load_flights(data_dir: Path, day: str) -> pd.DataFrame:
    f = pd.read_parquet(data_dir / day / "flights.parquet")
    ac_path = data_dir / "aircraft.parquet"
    if ac_path.exists():
        f = f.merge(pd.read_parquet(ac_path), on="icao24", how="left")
    for col in ("operator", "typecode", "model", "manufacturerName", "registration"):
        if col not in f.columns:
            f[col] = np.nan

    # heure de l'événement : décollage = firstSeen, atterrissage = lastSeen
    f["event_utc"] = f["first_seen_utc"].where(f["direction"] == "departure", f["last_seen_utc"])
    f["event_paris"] = f["event_utc"].dt.tz_convert("Europe/Paris")
    f["hour"] = f["event_paris"].dt.hour

    # nom de compagnie : opérateur le plus fréquent pour chaque préfixe de callsign
    names = (f.dropna(subset=["operator", "airline_code"])
             .groupby("airline_code")["operator"].agg(lambda s: s.mode().iat[0]))
    label = f["airline_code"].map(names)
    f["airline"] = np.where(
        f["airline_code"].isna(), "Non identifiée",
        np.where(label.notna(), label + " (" + f["airline_code"].fillna("") + ")",
                 f["airline_code"].fillna("")))

    f["aircraft_type"] = f["typecode"].fillna(f["model"]).fillna("Inconnu")
    f["flight_key"] = f["icao24"] + "_" + f["firstSeen"].astype(str) + "_" + f["direction"]
    return f


def load_tracks(data_dir: Path, day: str) -> pd.DataFrame:
    path = data_dir / day / "tracks.parquet"
    if not path.exists():
        return pd.DataFrame()
    t = pd.read_parquet(path)
    if t.empty:
        return t
    t = t.dropna(subset=["lat", "lon"]).sort_values(["flight_key", "time"]).reset_index(drop=True)

    t["alt_m"] = t["baro_altitude_m"]
    t["alt_ft"] = t["alt_m"] * M_TO_FT
    t["dist_bva_km"] = haversine_km(t["lat"], t["lon"], *BVA)

    # vitesse sol estimée entre deux points successifs d'un même vol
    g = t.groupby("flight_key")
    dt = g["time"].diff()
    d_km = haversine_km(g["lat"].shift(), g["lon"].shift(), t["lat"], t["lon"])
    speed = d_km / dt * 3600
    speed = speed.where((dt >= 1) & (dt <= 180) & (speed <= 1000))
    t["speed_kmh"] = speed.groupby(t["flight_key"]).transform(
        lambda s: s.rolling(5, center=True, min_periods=1).median())
    t["speed_kt"] = t["speed_kmh"] / 1.852

    t["phase"] = np.where(t["on_ground"], "Au sol",
                          np.where(t["direction"] == "arrival", "Approche", "Montée"))
    return t


def profile_by_distance(t: pd.DataFrame, max_km: int = 80, step_km: int = 5) -> pd.DataFrame:
    """Altitude et vitesse médianes selon la distance à BVA, par sens de trafic."""
    if t.empty:
        return t
    air = t[(~t["on_ground"]) & (t["dist_bva_km"] <= max_km)].copy()
    air["dist_bin"] = (air["dist_bva_km"] // step_km * step_km + step_km / 2)
    return (air.groupby(["direction", "dist_bin"])
            .agg(alt_ft=("alt_ft", "median"), speed_kt=("speed_kt", "median"),
                 n=("flight_key", "nunique"))
            .reset_index())
