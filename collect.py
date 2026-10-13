"""
AirTrack Beauvais - collecte locale d'une journée (OpenSky), équivalent du notebook Colab.

Usage (depuis le dossier du projet) :
    python collect.py --date 2026-10-08
    python collect.py --date 2026-10-08 --max-tracks 30        # essai léger
Options : --radius-nm 25  --ceiling-ft 8500  --data-dir data

Les fichiers sont écrits dans data/ (lu directement par l'app) et une archive
airtrack_data.zip est créée. Les identifiants sont demandés au lancement.
"""
import argparse
import getpass
import os
from pathlib import Path

import json, shutil, tempfile, time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

AIRPORT = "LFOB"
BVA = (49.4544, 2.1128)          # position de l'aéroport (lat, lon)
API_ROOT = "https://opensky-network.org/api"
TOKEN_URL = ("https://auth.opensky-network.org/auth/realms/opensky-network"
             "/protocol/openid-connect/token")
AIRCRAFT_DB_URL = "https://opensky-network.org/datasets/metadata/aircraftDatabase.csv"
M_TO_FT = 3.28084
NM_TO_KM = 1.852
MAX_WAIT_S = 600                 # au-delà, on arrête proprement (crédits épuisés)


class RateLimited(Exception):
    """Crédits OpenSky épuisés : la reprise se fait plus tard grâce au cache."""
    def __init__(self, wait):
        super().__init__(f"Limite OpenSky atteinte, nouvelle tentative possible dans {wait}s")
        self.wait = wait


class OpenSkyClient:
    """OAuth2 client credentials + renouvellement du token + réessais réseau + gestion du 429."""

    def __init__(self, client_id, client_secret):
        self.client_id, self.client_secret = client_id, client_secret
        self._token, self._expires_at = None, 0.0
        self.session = requests.Session()

    def _refresh_token(self):
        r = None
        for attempt in range(10):
            try:
                r = self.session.post(TOKEN_URL, data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id, "client_secret": self.client_secret,
                }, timeout=60)
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                wait = min(10 * (attempt + 1), 60)
                print(f"  authentification injoignable ({type(e).__name__}), essai {attempt + 1}/10, nouvel essai dans {wait}s")
                time.sleep(wait)
        if r is None:
            raise RuntimeError("Serveur d'authentification OpenSky injoignable après 10 essais : menu Exécution > Se déconnecter et supprimer l'environnement d'exécution, puis réessayez (ou attendez 30 min).")
        if r.status_code in (400, 401):
            raise SystemExit("Identifiants refusés par OpenSky : vérifiez client_id / client_secret")
        r.raise_for_status()
        payload = r.json()
        self._token = payload["access_token"]
        self._expires_at = time.time() + payload.get("expires_in", 300) - 30

    def get(self, path, params, max_retries=8):
        """GET authentifié. Renvoie le JSON, ou None si 404 (aucune donnée)."""
        for attempt in range(max_retries):
            if time.time() >= self._expires_at:
                self._refresh_token()
            try:
                r = self.session.get(f"{API_ROOT}{path}", params=params,
                                     headers={"Authorization": f"Bearer {self._token}"}, timeout=60)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                wait = min(10 * (attempt + 1), 60)
                print(f"  erreur réseau ({type(e).__name__}), nouvel essai dans {wait}s")
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return None
            if r.status_code == 401:
                self._expires_at = 0
                continue
            if r.status_code == 429:
                wait = int(r.headers.get("X-Rate-Limit-Retry-After-Seconds", 60))
                if wait > MAX_WAIT_S:
                    raise RateLimited(wait)
                print(f"  limite atteinte : pause de {wait}s")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            remaining = r.headers.get("X-Rate-Limit-Remaining")
            if remaining is not None:
                self.credits_remaining = remaining
            return r.json()
        raise RuntimeError(f"Échec de la requête {path} {params}")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def day_bounds_paris(date_str):
    """Début et fin (timestamps UTC) de la journée LOCALE de Paris (gère l'heure d'été)."""
    tz = ZoneInfo("Europe/Paris")
    d0 = datetime.strptime(date_str, "%Y-%m-%d")
    return (int(d0.replace(tzinfo=tz).timestamp()),
            int((d0 + timedelta(days=1)).replace(tzinfo=tz).timestamp()))


# ------------------------------------------------------------------ vols
def fetch_flights(client, begin, end, chunk_hours=6):
    """Arrivées et départs de LFOB dont l'événement (atterrissage / décollage) tombe dans [begin, end[."""
    rows = []
    for direction in ("arrival", "departure"):
        t = begin
        while t < end:
            t2 = min(t + chunk_hours * 3600, end)
            print(f"{direction:9s} {datetime.utcfromtimestamp(t):%d/%m %H:%M} -> "
                  f"{datetime.utcfromtimestamp(t2):%d/%m %H:%M} UTC")
            for f in client.get(f"/flights/{direction}", {"airport": AIRPORT, "begin": t, "end": t2}) or []:
                f["direction"] = direction
                rows.append(f)
            t = t2
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["callsign"] = df["callsign"].fillna("").str.strip()
    df["airline_code"] = df["callsign"].str.extract(r"^([A-Z]{3})\d", expand=False)
    df["first_seen_utc"] = pd.to_datetime(df["firstSeen"], unit="s", utc=True)
    df["last_seen_utc"] = pd.to_datetime(df["lastSeen"], unit="s", utc=True)
    df = df.drop_duplicates(subset=["icao24", "firstSeen", "direction"])
    event = df["firstSeen"].where(df["direction"] == "departure", df["lastSeen"])
    df = df[(event >= begin) & (event < end)].reset_index(drop=True)
    df["flight_key"] = df["icao24"] + "_" + df["firstSeen"].astype(str) + "_" + df["direction"]
    return df


# ------------------------------------------------------------ trajectoires
TRACK_COLS = ["icao24", "callsign", "direction", "flight_key", "time",
              "lat", "lon", "baro_altitude_m", "true_track", "on_ground"]


def fetch_tracks(client, flights, cache_dir, max_tracks=None):
    """Trajectoires vol par vol, avec cache disque : une interruption ne fait rien perdre."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = flights if max_tracks is None else flights.head(max_tracks)
    rows, stats = [], {"avec_trajectoire": 0, "sans_trajectoire": 0, "non_tentes": 0}
    for i, f in enumerate(todo.itertuples(), start=1):
        cache_file = cache_dir / f"{f.flight_key}.json"
        if cache_file.exists():
            track = json.loads(cache_file.read_text())
        else:
            mid = (f.firstSeen + f.lastSeen) // 2          # doit tomber dans la durée du vol
            try:
                track = client.get("/tracks", {"icao24": f.icao24, "time": mid}) or {}
            except RateLimited as e:
                print(f"⏸️  {e}. Relancez plus tard : les {i - 1} trajectoires déjà reçues sont en cache.")
                stats["non_tentes"] = len(todo) - (i - 1)
                break
            cache_file.write_text(json.dumps(track))
        path = track.get("path") or []
        stats["avec_trajectoire" if path else "sans_trajectoire"] += 1
        for p in path:   # [time, lat, lon, baro_altitude, true_track, on_ground]
            rows.append((f.icao24, f.callsign, f.direction, f.flight_key, *p))
        if i % 25 == 0 or i == len(todo):
            print(f"trajectoires : {i}/{len(todo)}  (crédits restants : "
                  f"{getattr(client, 'credits_remaining', '?')})")
    df = pd.DataFrame(rows, columns=TRACK_COLS)
    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df, stats


# ------------------------------------------------------------ zone d'étude
def point_in_polygon(lon, lat, poly):
    """Algorithme du rayon, vectorisé. poly : tableau (n, 2) de (lon, lat)."""
    inside = np.zeros(len(lon), dtype=bool)
    j = len(poly) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            cross = ((yi > lat) != (yj > lat)) & (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi)
            inside ^= cross
            j = i
    return inside


def inside_zone(lat, lon, alt_m, on_ground, zone):
    """Vrai pour les points dans la zone : limites latérales ET limite supérieure."""
    lat = pd.to_numeric(pd.Series(lat), errors="coerce").to_numpy(float)
    lon = pd.to_numeric(pd.Series(lon), errors="coerce").to_numpy(float)
    if zone.get("polygon"):
        poly = np.array([(lo, la) for la, lo in zone["polygon"]], dtype=float)
        lateral = point_in_polygon(lon, lat, poly)
    else:
        lateral = haversine_km(lat, lon, *BVA) <= zone["radius_nm"] * NM_TO_KM
    lateral &= ~np.isnan(lat) & ~np.isnan(lon)
    if zone.get("ceiling_ft") is None:
        return lateral
    alt_ft = pd.to_numeric(pd.Series(alt_m), errors="coerce").to_numpy(float) * M_TO_FT
    ground = pd.Series(on_ground).fillna(False).to_numpy(bool)
    return lateral & (np.isnan(alt_ft) | ground | (alt_ft <= zone["ceiling_ft"]))


# ------------------------------------------------------------ aéronefs
def build_aircraft_table(icao24s, csv_path=None):
    """Modèle / compagnie de chaque avion, via la base d'aéronefs OpenSky (échec = table vide)."""
    cols = ["icao24", "registration", "manufacturerName", "model", "typecode", "operator", "operatorIcao"]
    try:
        if csv_path is None:
            csv_path = Path(tempfile.gettempdir()) / "aircraftDatabase.csv"
            if not csv_path.exists():
                print("Téléchargement de la base d'aéronefs (~100 Mo)...")
                with requests.get(AIRCRAFT_DB_URL, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(csv_path, "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
        db = pd.read_csv(csv_path, usecols=lambda c: c in cols, dtype=str)
        return db[db["icao24"].isin(set(icao24s))].drop_duplicates("icao24")
    except Exception as e:
        print(f"⚠️ Base d'aéronefs indisponible ({type(e).__name__}) : types d'appareils non renseignés.")
        return pd.DataFrame(columns=cols)


# ------------------------------------------------------------ pipeline complet
def run_pipeline(client, date_local, max_tracks, zone, data_dir):
    out = Path(data_dir) / date_local
    out.mkdir(parents=True, exist_ok=True)
    begin, end = day_bounds_paris(date_local)

    flights = fetch_flights(client, begin, end)
    if flights.empty:
        raise RuntimeError("Aucun vol renvoyé : essayez une date plus ancienne (J-2 ou plus).")
    print(f"\n{len(flights)} mouvements LFOB le {date_local} : "
          f"{(flights.direction == 'arrival').sum()} arrivées, {(flights.direction == 'departure').sum()} départs\n")

    aircraft = build_aircraft_table(flights["icao24"].unique())
    aircraft.to_parquet(Path(data_dir) / "aircraft.parquet", index=False)
    print(f"{len(aircraft)} aéronefs identifiés sur {flights['icao24'].nunique()}\n")

    tracks, stats = fetch_tracks(client, flights, out / "tracks_cache", max_tracks)
    tracks["alt_ft"] = pd.to_numeric(tracks["baro_altitude_m"], errors="coerce") * M_TO_FT
    tracks["in_zone"] = inside_zone(tracks["lat"], tracks["lon"], tracks["baro_altitude_m"],
                                    tracks["on_ground"], zone) if len(tracks) else pd.Series(dtype=bool)

    n_all = tracks.groupby("flight_key").size()
    n_zone = tracks[tracks["in_zone"]].groupby("flight_key").size()
    flights["has_track"] = flights["flight_key"].isin(n_all.index)
    flights["n_zone_points"] = flights["flight_key"].map(n_zone).fillna(0).astype(int)
    flights["in_zone"] = flights["n_zone_points"] > 0

    tracks_zone = tracks[tracks["in_zone"]].drop(columns=["alt_ft"]).reset_index(drop=True)
    flights.to_parquet(out / "flights.parquet", index=False)
    tracks_zone.to_parquet(out / "tracks.parquet", index=False)

    meta = {"date_local": date_local, "airport": AIRPORT, "flights": int(len(flights)),
            "flights_with_track": int(flights["has_track"].sum()),
            "flights_in_zone": int(flights["in_zone"].sum()),
            "track_points_in_zone": int(len(tracks_zone)), "track_stats": stats,
            "zone": ("contour personnalisé" if zone.get("polygon") else f"cercle {zone['radius_nm']} NM"),
            "plafond_ft": zone.get("ceiling_ft"),
            "collected_at": datetime.utcnow().isoformat() + "Z"}
    (out / "zone.json").write_text(json.dumps({**zone, "center": BVA}, indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return flights, tracks, tracks_zone, aircraft, meta


def make_zip(data_dir, date_local, zip_base="airtrack_data"):
    """Archive prête à déposer sur GitHub : data/aircraft.parquet + data/<date>/..."""
    tmp = Path("export_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "data" / date_local).mkdir(parents=True)
    shutil.copy(Path(data_dir) / "aircraft.parquet", tmp / "data")
    for name in ("flights.parquet", "tracks.parquet", "zone.json", "meta.json"):
        p = Path(data_dir) / date_local / name
        if p.exists():
            shutil.copy(p, tmp / "data" / date_local)
    for p in (tmp / "data").rglob("*"):
        if p.is_file() and p.stat().st_size > 25 * 1024 * 1024:
            print(f"⚠️ {p.name} fait {p.stat().st_size / 1e6:.0f} Mo (> 25 Mo : trop gros pour l'envoi GitHub par navigateur)")
    return shutil.make_archive(zip_base, "zip", tmp)


def main():
    ap = argparse.ArgumentParser(description="Collecte OpenSky pour Paris-Beauvais (LFOB)")
    ap.add_argument("--date", default="2026-10-08", help="Jour LOCAL (Paris) à collecter, AAAA-MM-JJ")
    ap.add_argument("--max-tracks", type=int, default=None, help="Limiter le nombre de trajectoires")
    ap.add_argument("--radius-nm", type=float, default=25, help="Rayon de la zone autour de LFOB (NM)")
    ap.add_argument("--ceiling-ft", type=float, default=8500, help="Plafond de la zone (ft)")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    cid = os.getenv("OPENSKY_CLIENT_ID") or input("client_id : ").strip()
    secret = os.getenv("OPENSKY_CLIENT_SECRET") or getpass.getpass("client_secret (invisible) : ").strip()

    zone = {"polygon": None, "radius_nm": args.radius_nm, "ceiling_ft": args.ceiling_ft}
    args.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Zone d'étude : cercle de {args.radius_nm} NM autour de LFOB | plafond {args.ceiling_ft} ft")

    client = OpenSkyClient(cid, secret)
    _, _, _, _, meta = run_pipeline(client, args.date, args.max_tracks, zone, args.data_dir)

    print("\n===== BILAN =====")
    for k, v in meta.items():
        print(f"{k:22s} {v}")
    if meta["track_stats"]["non_tentes"]:
        print("\nCollecte interrompue (crédits) : relancez la même commande plus tard, le cache reprend.")
    print("\nArchive créée :", make_zip(args.data_dir, args.date))


if __name__ == "__main__":
    main()
