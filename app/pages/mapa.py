import streamlit as st
import pandas as pd
import folium
import openrouteservice
from openrouteservice import convert
from folium.plugins import MarkerCluster
from collections import OrderedDict

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
# Origen fijo (actualizado)
# -------------------------
origen = (41.544608, 2.441753)  # (lat, lon) nuevo punto origen solicitado
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
all_stops = [{"index": None, "name": "Origen - Plaça Catalunya", "lat": origen[0], "lon": origen[1], "city": None}]

for idx, row in df.iterrows():
    parsed = parse_latlon(row.get("coordenadas_gps", ""))
    # Intentamos leer el campo 'Ciudad' con varias capitalizaciones comunes
    city_value = row.get("Ciudad", row.get("ciudad", row.get("CIUDAD", None)))
    display_name = row.get("nombre_completo", f"Destino {idx}")
    if parsed:
        lat, lon = parsed
        all_coords.append([lon, lat])
        all_stops.append({
            "index": idx,
            "name": display_name,
            "lat": lat,
            "lon": lon,
            "city": city_value
        })
    else:
        st.warning(f"Fila {idx} - coordenadas no parseables: {row.get('coordenadas_gps')} - Ciudad: {city_value}")
        # También añadimos como stop 'sin coord' para poder informar después si hace falta
        all_stops.append({
            "index": idx,
            "name": display_name,
            "lat": None,
            "lon": None,
            "city": city_value
        })

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
    # Si las coordenadas eran None (parseo fallido) ya las ignoramos aquí
    if candidate is None or candidate[0] is None:
        # Registrar como descartado por falta de coordenadas
        discarded_stops.append({
            "index": all_stops[i].get("index"),
            "name": all_stops[i].get("name"),
            "city": all_stops[i].get("city"),
            "lat": all_stops[i].get("lat"),
            "lon": all_stops[i].get("lon"),
            "error": "Coordenadas no parseables"
        })
        st.warning(f"Descartado (coordenadas no parseables): {all_stops[i]['name']} — Ciudad: {all_stops[i].get('city')}")
        continue

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
            "city": all_stops[i].get("city"),
            "lat": all_stops[i].get("lat"),
            "lon": all_stops[i].get("lon"),
            "error": str(e)
        })
        st.warning(f"Descartado (no enrutable): {all_stops[i]['name']} — Ciudad: {all_stops[i].get('city')} — {all_stops[i]['lat']}, {all_stops[i]['lon']}")

# Si solo queda el origen, no hay ruta posible
if len(valid_coords) < 2:
    st.error("No hay suficientes puntos enrutable (necesitamos al menos el origen y un destino válido).")
    if discarded_stops:
        st.markdown("### Destinos descartados")
        st.dataframe(pd.DataFrame(discarded_stops))
        # Lista clara con nombres de la columna Ciudad (o nombre si Ciudad no existe)
        st.markdown("#### Ciudades de los destinos descartados")
        for d in discarded_stops:
            display_city = d.get("city") or d.get("name") or "Sin nombre"
            st.markdown(f"- **{display_city}** (Destino: {d.get('name')}, motivo: {d.get('error')})")
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
    # Si existe ciudad, mostrar solo la ciudad en el popup/tooltip;
    # si no, mostrar índice + nombre
    if s.get("city"):
        popup_text = f"{s['city']}"
        tooltip_text = f"{s['city']}"
    else:
        popup_text = f"{i}. {s['name']}"
        tooltip_text = popup_text

    folium.Marker(
        location=[s["lat"], s["lon"]],
        popup=popup_text,
        tooltip=tooltip_text,
        icon=folium.DivIcon(
            html=f"""
            <div style="
                display:flex;
                align-items:center;
                justify-content:center;
                width:24px;
                height:24px;
                background:#d9534f;
                color:white;
                border-radius:50%;
                font-size:12px;
                font-weight:bold;
                border:2px solid white;
                box-sizing:border-box;
            ">{i}</div>
            """
        )
    ).add_to(marker_cluster)

# Marcadores para descartados (gris)
for d in discarded_stops:
    popup_text = f"Descartado: {d.get('name')}"
    if d.get("city"):
        popup_text += f" — Ciudad: {d.get('city')}"
    folium.Marker(
        location=[d["lat"], d["lon"]] if d.get("lat") is not None else [center_lat, -3.5],
        popup=popup_text,
        tooltip=popup_text,
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

# -------------------------
# Mostrar mapa
# -------------------------
st.markdown("### Mapa con la ruta completa (se muestran solo los destinos válidos)")
st.components.v1.html(m._repr_html_(), width=1100, height=700)

# -------------------------
# Mostrar NOMBRES (campo ciudad) de las ciudades descartadas — justo debajo del mapa
# -------------------------
cities = [d.get("city") or d.get("name") or "Sin nombre" for d in discarded_stops]
unique_cities = list(OrderedDict.fromkeys(cities))

if discarded_stops and unique_cities:
    st.markdown("### Ciudades de los destinos descartados")
    st.write(f"Se han descartado **{len(discarded_stops)}** destinos — **{len(unique_cities)}** ciudades únicas:")
    for c in unique_cities:
        st.markdown(f"- **{c}**")
else:
    st.info("No hay destinos descartados. Todos los puntos son enrutable.")
