import streamlit as st
import pandas as pd
import folium
import openrouteservice
from openrouteservice import convert
from folium.plugins import MarkerCluster

st.set_page_config(page_title="Mapa Destinos (Ruta por todos los puntos)", layout="wide")
st.title("Mapa de Destinos + Ruta desde Barcelona (pasando por todas las paradas del CSV)")

# -------------------------
# Configuración / carga CSV
# -------------------------
@st.cache_data
def load_destinos(path):
    return pd.read_csv(path)

CSV_PATH = r"C:\Users\savit\Desktop\Master IABiGData\Proyecto 1 Optimización Logistica\PROTOTYPE-IA-DELIVERY\data\destinos_geocodificados.csv"
df = load_destinos(CSV_PATH)

st.subheader("Destinos disponibles (CSV)")
st.dataframe(df)

# -------------------------
# Origen fijo
# -------------------------
origen = (41.3870, 2.1690)  # (lat, lon) Plaça Catalunya
origen_lonlat = [origen[1], origen[0]]  # [lon, lat] para ORS

# -------------------------
# Preparar lista de coordenadas (todas) y parsing defensivo
# -------------------------
def parse_latlon(text):
    if pd.isna(text):
        return None
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) < 2:
        return None
    try:
        lat = float(parts[0])
        lon = float(parts[1])
        return lat, lon
    except ValueError:
        return None

# Construcción inicial de paradas desde CSV (sin validar enrutable aún)
all_coords = [[origen[1], origen[0]]]  # lista de [lon, lat] empezando por origen
all_stops = [{"index": None, "name": "Origen - Plaça Catalunya", "lat": origen[0], "lon": origen[1]}]

for idx, row in df.iterrows():
    parsed = parse_latlon(row.get("coordenadas_gps", ""))
    if parsed:
        lat, lon = parsed
        all_coords.append([lon, lat])
        all_stops.append({
            "index": idx,
            "name": row.get("nombre_completo", f"Destino {idx}"),
            "lat": lat,
            "lon": lon
        })
    else:
        st.warning(f"Fila {idx} - coordenadas no parseables: {row.get('coordenadas_gps')}")

if len(all_coords) < 2:
    st.error("No hay destinos válidos en el CSV para construir una ruta.")
    st.stop()

# -------------------------
# Cliente ORS (clave en claro según petición)
# -------------------------
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjU1NWQ1MGYzZTMzZjQ4YjJiMmU5MWU0MjczZTc2MWI4IiwiaCI6Im11cm11cjY0In0="
client = openrouteservice.Client(key=ORS_API_KEY)

# -------------------------
# Validación punto a punto para descartar los no enrutable
# -------------------------
st.info("Validando destinos contra la red viaria (se descartarán los no enrutable)...")
radiuses_check = 1000  # metros; ajustar si se desea menos/más tolerancia

valid_coords = [all_coords[0]]   # puntos aceptados para la ruta final (inician con origen)
valid_stops = [all_stops[0]]     # stops correspondientes a valid_coords
discarded_stops = []             # stops descartados con error

# Intentar conectar secuencialmente desde el último válido hasta el candidato
for i in range(1, len(all_coords)):
    candidate = all_coords[i]
    try:
        # Petición corta solo para validar enrutable entre last_valid y candidate
        _ = client.directions(
            coordinates=[valid_coords[-1], candidate],
            profile="driving-car",
            radiuses=[radiuses_check, radiuses_check],
            format="json"
        )
        # Si no lanza excepción, consideramos el punto válido
        valid_coords.append(candidate)
        valid_stops.append(all_stops[i])
    except Exception as e:
        # Guardamos información del descartado para mostrarla luego
        discarded_stops.append({
            "index": all_stops[i].get("index"),
            "name": all_stops[i].get("name"),
            "lat": all_stops[i].get("lat"),
            "lon": all_stops[i].get("lon"),
            "error": str(e)
        })
        st.warning(f"Descartado (no enrutable): {all_stops[i]['name']} — {all_stops[i]['lat']}, {all_stops[i]['lon']}")

# Si solo queda el origen, no hay ruta posible
if len(valid_coords) < 2:
    st.error("No hay suficientes puntos enrutable (necesitamos al menos el origen y un destino válido).")
    if discarded_stops:
        st.markdown("### Destinos descartados")
        st.dataframe(pd.DataFrame(discarded_stops))
    st.stop()

# -------------------------
# Mapa base y marcadores (solo para puntos validados y descartados)
# -------------------------
# Centrar mapa en el primer destino válido si existe
center_lat = valid_stops[1]["lat"] if len(valid_stops) > 1 else origen[0]
m = folium.Map(location=[center_lat, -3.5], zoom_start=6)
marker_cluster = MarkerCluster().add_to(m)

# Marcador origen
folium.Marker(
    location=[origen[0], origen[1]],
    popup="Origen: Plaça Catalunya",
    tooltip="Origen",
    icon=folium.Icon(color="blue", icon="home")
).add_to(m)

# Marcadores para paradas válidas (numeradas)
for i, s in enumerate(valid_stops[1:], start=1):
    folium.Marker(
        location=[s["lat"], s["lon"]],
        popup=f"{i}. {s['name']}",
        tooltip=f"{i}. {s['name']}",
        icon=folium.DivIcon(html=f"""<div style="font-size:12px;color:white;background:#d9534f;border-radius:12px;padding:4px 6px;">{i}</div>""")
    ).add_to(marker_cluster)

# Marcadores para descartados (gris)
for d in discarded_stops:
    folium.Marker(
        location=[d["lat"], d["lon"]],
        popup=f"Descartado: {d['name']}",
        tooltip=f"Descartado: {d['name']}",
        icon=folium.Icon(color="gray", icon="remove")
    ).add_to(m)

# -------------------------
# Solicitar la ruta final con los puntos validados
# -------------------------
with st.spinner("Calculando ruta final con los destinos válidos..."):
    try:
        radiuses_final = [radiuses_check] * len(valid_coords)
        route = client.directions(
            coordinates=valid_coords,
            profile="driving-car",
            format="json",
            radiuses=radiuses_final,
            optimize_waypoints=False
        )
    except Exception as e:
        st.error(f"Error al solicitar la ruta a ORS (final): {e}")
        st.stop()

# -------------------------
# Decodificar geometría y pintar polilínea
# -------------------------
geometry = route["routes"][0]["geometry"]
decoded = convert.decode_polyline(geometry)["coordinates"]  # devuelve lista [lon, lat]
ruta_latlon = [(lat, lon) for lon, lat in decoded]  # convertir a (lat, lon) para folium

folium.PolyLine(
    ruta_latlon,
    color="blue",
    weight=5,
    opacity=0.8
).add_to(m)

# -------------------------
# Resumen de distancia / tiempos y distancias por tramo
# -------------------------
summary = route["routes"][0].get("summary", {})
distancia_total_m = summary.get("distance", None)
duracion_total_s = summary.get("duration", None)

segments = route["routes"][0].get("segments", [])

legs = []
stop_names = [s["name"] for s in valid_stops]

if segments and len(segments) == (len(stop_names) - 1):
    for i, seg in enumerate(segments):
        legs.append({
            "from": stop_names[i],
            "to": stop_names[i+1],
            "distance_km": seg.get("distance", 0) / 1000.0,
            "duration_min": seg.get("duration", 0) / 60.0
        })
else:
    if distancia_total_m is not None:
        approx = distancia_total_m / max(1, (len(stop_names) - 1)) / 1000.0
        for i in range(len(stop_names) - 1):
            legs.append({
                "from": stop_names[i],
                "to": stop_names[i+1],
                "distance_km": approx,
                "duration_min": None
            })

st.markdown("### Resumen de ruta")
if distancia_total_m is not None:
    st.write(f"Distancia total aproximada: **{distancia_total_m/1000:.2f} km**")
if duracion_total_s is not None:
    st.write(f"Duración total aproximada: **{duracion_total_s/3600:.2f} horas**")

df_legs = pd.DataFrame(legs)
st.dataframe(df_legs)

# Mostrar tabla de descartados (si los hay)
if discarded_stops:
    st.markdown("### Destinos descartados por no ser enrutable (no aparecerán en la ruta)")
    st.dataframe(pd.DataFrame(discarded_stops))

# -------------------------
# Mostrar mapa
# -------------------------
st.markdown("### Mapa con la ruta completa (se muestran solo los destinos válidos y los descartados)")
st.components.v1.html(m._repr_html_(), width=1100, height=700)
