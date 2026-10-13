"""AirTrack Beauvais - app Streamlit.   Lancer :  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from process import BVA, list_days, load_flights, load_tracks, profile_by_distance

DATA_DIR = Path(__file__).parent / "data"
COLORS = {"arrival": [30, 120, 255], "departure": [255, 130, 0]}
LABELS = {"arrival": "Arrivées", "departure": "Départs"}

st.set_page_config(page_title="AirTrack Beauvais", page_icon="✈️", layout="wide")


@st.cache_data
def get_data(day: str):
    return load_flights(DATA_DIR, day), load_tracks(DATA_DIR, day)


@st.cache_data
def get_profile(day: str):
    return profile_by_distance(get_data(day)[1])


days = list_days(DATA_DIR) if DATA_DIR.exists() else []
if not days:
    st.error("Aucune donnée trouvée dans le dossier `data/`. Lancez d'abord `collect.py` "
             "(ou copiez ici le dossier `data/` téléchargé depuis Colab).")
    st.stop()

st.title("✈️ AirTrack Beauvais")
st.caption("Une journée de trafic aérien à Paris-Beauvais (LFOB) · données OpenSky Network")

with st.sidebar:
    day = st.selectbox("Journée", days, index=len(days) - 1)
    flights, tracks = get_data(day)
    directions = st.multiselect("Sens du trafic", list(LABELS), default=list(LABELS),
                                format_func=LABELS.get)
    top_airlines = flights["airline"].value_counts().index.tolist()
    airlines = st.multiselect("Compagnies (vide = toutes)", top_airlines)

f = flights[flights["direction"].isin(directions)]
if airlines:
    f = f[f["airline"].isin(airlines)]
t = tracks[tracks["flight_key"].isin(f["flight_key"])] if not tracks.empty else tracks

n_arr, n_dep = (f["direction"] == "arrival").sum(), (f["direction"] == "departure").sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mouvements", len(f))
c2.metric("Arrivées / Départs", f"{n_arr} / {n_dep}")
c3.metric("Compagnies", f["airline"].nunique())
c4.metric("Trajectoires disponibles", t["flight_key"].nunique() if not t.empty else 0)
if not tracks.empty and tracks["flight_key"].nunique() < len(flights):
    st.info(f"Trajectoires récupérées pour {tracks['flight_key'].nunique()} vols sur "
            f"{len(flights)} : la carte et les profils portent sur cet échantillon. "
            "Relancez la collecte sans `--max-tracks` pour tout avoir.")

tab_map, tab_hour, tab_prof, tab_top = st.tabs(
    ["🗺️ Carte", "🕐 Trafic par heure", "📈 Altitudes & vitesses", "🏢 Compagnies & appareils"])

# ---------------------------------------------------------------- Carte
with tab_map:
    if t.empty:
        st.warning("Pas de trajectoires pour cette sélection.")
    else:
        paths = (t.groupby(["flight_key", "callsign", "direction"])
                 .apply(lambda g: [[r.lon, r.lat] for r in g.itertuples()], include_groups=False)
                 .rename("path").reset_index())
        paths["color"] = paths["direction"].map(COLORS)
        deck = pdk.Deck(
            layers=[
                pdk.Layer("PathLayer", paths, get_path="path", get_color="color",
                          width_min_pixels=2, pickable=True, opacity=0.7),
                pdk.Layer("ScatterplotLayer", pd.DataFrame([{"lon": BVA[1], "lat": BVA[0]}]),
                          get_position="[lon, lat]", get_radius=1500, get_fill_color=[220, 20, 60]),
            ],
            initial_view_state=pdk.ViewState(latitude=BVA[0], longitude=BVA[1], zoom=7.5),
            tooltip={"text": "{callsign} ({direction})"},
        )
        st.pydeck_chart(deck)
        st.caption("🔵 arrivées · 🟠 départs · 🔴 aéroport de Beauvais")

# ---------------------------------------------------------------- Trafic horaire
with tab_hour:
    hourly = (f.groupby(["hour", "direction"]).size().rename("vols").reset_index())
    hourly["Sens"] = hourly["direction"].map(LABELS)
    fig = px.bar(hourly, x="hour", y="vols", color="Sens", barmode="stack",
                 color_discrete_map={"Arrivées": "#1e78ff", "Départs": "#ff8200"},
                 labels={"hour": "Heure locale (Paris)", "vols": "Mouvements"})
    fig.update_xaxes(dtick=1, range=[-0.5, 23.5])
    st.plotly_chart(fig, width="stretch")
    if not f.empty:
        peak = f.groupby("hour").size()
        st.write(f"**Pic d'activité :** {peak.idxmax()} h - {peak.idxmax() + 1} h "
                 f"({peak.max()} mouvements).")

# ---------------------------------------------------------------- Profils
with tab_prof:
    prof = get_profile(day)
    if prof.empty:
        st.warning("Pas de trajectoires pour calculer les profils.")
    else:
        prof = prof[prof["direction"].isin(directions)].copy()
        prof["Sens"] = prof["direction"].map(LABELS)
        cmap = {"Arrivées": "#1e78ff", "Départs": "#ff8200"}
        a, b = st.columns(2)
        fa = px.line(prof, x="dist_bin", y="alt_ft", color="Sens", markers=True,
                     color_discrete_map=cmap,
                     labels={"dist_bin": "Distance à BVA (km)", "alt_ft": "Altitude médiane (ft)"},
                     title="Altitude selon la distance à l'aéroport")
        a.plotly_chart(fa, width="stretch")
        fs = px.line(prof, x="dist_bin", y="speed_kt", color="Sens", markers=True,
                     color_discrete_map=cmap,
                     labels={"dist_bin": "Distance à BVA (km)", "speed_kt": "Vitesse sol médiane (kt)"},
                     title="Vitesse sol selon la distance à l'aéroport")
        b.plotly_chart(fs, width="stretch")
        st.caption("Médianes par tranche de 5 km, sur les vols en l'air. La vitesse est une "
                   "vitesse sol estimée à partir des positions successives (non fournie par OpenSky).")

# ---------------------------------------------------------------- Compagnies & appareils
with tab_top:
    n = st.slider("Nombre d'éléments affichés", 5, 20, 10)
    a, b = st.columns(2)
    top_a = f["airline"].value_counts().head(n).rename_axis("Compagnie").reset_index(name="Vols")
    a.plotly_chart(px.bar(top_a, x="Vols", y="Compagnie", orientation="h",
                          title="Top compagnies").update_yaxes(autorange="reversed"),
                   width="stretch")
    top_t = f["aircraft_type"].value_counts().head(n).rename_axis("Type").reset_index(name="Vols")
    b.plotly_chart(px.bar(top_t, x="Vols", y="Type", orientation="h",
                          title="Top types d'appareils").update_yaxes(autorange="reversed"),
                   width="stretch")
    unknown = (f["aircraft_type"] == "Inconnu").mean() * 100 if len(f) else 0
    st.caption(f"{unknown:.0f} % des vols sans type d'appareil identifié (absents de la base OpenSky).")
