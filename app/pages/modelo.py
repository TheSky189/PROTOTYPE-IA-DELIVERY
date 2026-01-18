import streamlit as st
import pandas as pd
import folium
import openrouteservice
from openrouteservice import convert
from folium.plugins import MarkerCluster
import colorsys
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import itertools
import copy
from typing import List, Dict

st.set_page_config(page_title="Asignación RF - Minimizar km (prioridad por tiempo)", layout="wide")
st.title("Asignación de Pedidos a Camiones y Calculo de Rutas")

ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjU1NWQ1MGYzZTMzZjQ4YjJiMmU5MWU0MjczZTc2MWI4IiwiaCI6Im11cm11cjY0In0="
try:
    client = openrouteservice.Client(key=ORS_API_KEY)
except Exception:
    client = None

REQUIRED_DESTINOS = {"DestinoID", "coordenadas_gps", "ciudad"}
REQUIRED_PEDIDOS = {"PedidoID", "DestinoEntregaID"}
REQUIRED_LINEAS = {"PedidoID", "ProductoID", "Cantidad"}
REQUIRED_PRODUCTOS = {"ProductoID", "TiempoFabricacionMedia", "Caducidad"}

ORIGEN = (41.544608, 2.441753)
ORIGEN_LONLAT = [ORIGEN[1], ORIGEN[0]]

def cargar_csv(f):
    return pd.read_csv(f) if f is not None else None

def validar(df, required, nombre):
    if not required.issubset(df.columns):
        st.error(f"{nombre}: faltan columnas {required - set(df.columns)}")
        st.stop()

def parse_latlon(text):
    try:
        lat, lon = map(float, str(text).split(","))
        return lat, lon
    except Exception:
        return None

def hsv_to_hex(h, s=0.75, v=0.9):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def nearest_neighbor_order(origen, stops: List[Dict]):
    remaining = stops.copy()
    ordered = []
    current_lat, current_lon = origen
    while remaining:
        nearest = min(remaining, key=lambda s: haversine(current_lat, current_lon, s["lat"], s["lon"]))
        ordered.append(nearest)
        remaining.remove(nearest)
        current_lat, current_lon = nearest["lat"], nearest["lon"]
    return ordered

def estimate_marginal_km(last_lat, last_lon, order_lat, order_lon):
    if any(pd.isna(x) for x in [last_lat, last_lon, order_lat, order_lon]):
        return 9999.0
    d_last_origin = haversine(last_lat, last_lon, ORIGEN[0], ORIGEN[1])
    d_last_order = haversine(last_lat, last_lon, order_lat, order_lon)
    d_order_origin = haversine(order_lat, order_lon, ORIGEN[0], ORIGEN[1])
    marginal = (d_last_order + d_order_origin) - d_last_origin
    return max(marginal, 0.0)

def route_distance_for_stops(stops: List[Dict]):
    if not stops:
        return 0.0
    ordered = nearest_neighbor_order(ORIGEN, stops)
    total = 0.0
    cur_lat, cur_lon = ORIGEN
    for s in ordered:
        total += haversine(cur_lat, cur_lon, s["lat"], s["lon"])
        cur_lat, cur_lon = s["lat"], s["lon"]
    total += haversine(cur_lat, cur_lon, ORIGEN[0], ORIGEN[1])
    return total

def two_opt_route_order(origen, stops: List[Dict], max_iter=200):
    if not stops:
        return []
    best = nearest_neighbor_order(origen, stops)
    best_distance = route_distance_for_stops(best)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        n = len(best)
        for i in range(n - 1):
            for k in range(i + 1, n):
                new_route = best[:i] + list(reversed(best[i:k+1])) + best[k+1:]
                new_distance = route_distance_for_stops(new_route)
                if new_distance + 1e-6 < best_distance:
                    best = new_route
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
    return best

def safe_to_numeric(s):
    try:
        return float(s)
    except Exception:
        return 0.0

st.header("1) Subir datasets (4 CSV obligatorios)")

c1, c2, c3, c4 = st.columns(4)
with c1:
    destinos_file = st.file_uploader("Destinos.csv", type="csv")
with c2:
    pedidos_file = st.file_uploader("Pedidos.csv", type="csv")
with c3:
    lineas_file = st.file_uploader("LineasPedido.csv", type="csv")
with c4:
    productos_file = st.file_uploader("Productos.csv", type="csv")

destinos_df = cargar_csv(destinos_file)
pedidos_df = cargar_csv(pedidos_file)
lineas_df = cargar_csv(lineas_file)
productos_df = cargar_csv(productos_file)

def algun_df_vacio(*dfs):
    return any(df is None or df.empty for df in dfs)

if algun_df_vacio(destinos_df, pedidos_df, lineas_df, productos_df):
    st.info("Sube los 4 CSV para poder entrenar y asignar.")
    st.stop()

validar(destinos_df, REQUIRED_DESTINOS, "Destinos")
validar(pedidos_df, REQUIRED_PEDIDOS, "Pedidos")
validar(lineas_df, REQUIRED_LINEAS, "Líneas de Pedido")
validar(productos_df, REQUIRED_PRODUCTOS, "Productos")

st.success("CSV cargados y validados.")

st.sidebar.subheader("Configuración de las variables")

capacidad_camion = st.sidebar.number_input("Capacidad del camión (unidades)",min_value=1,max_value=5000,value=500,step=1)
velocidad_media = st.sidebar.number_input("Velocidad media (km/h)",min_value=5.0,max_value=200.0,value=80.0,step=1.0)

FACTOR_CONSUMO = 25.0 / 80.0
consumo_l_100km = velocidad_media * FACTOR_CONSUMO

precio_litro = st.sidebar.number_input("Precio combustible (€/L)",min_value=0.1,max_value=5.0,value=1.65,step=0.01)
st.sidebar.number_input("Consumo estimado (L / 100 km)",value=round(consumo_l_100km, 2),disabled=True)

st.header("2) Parámetros para el cálculo")

p1, p2 = st.columns(2)

with p1:
    n_clusters = 8
    st.number_input(
        "Número de clusters geográficos (zonas)",
        value=n_clusters,
        disabled=True
    )

with p2:
    rf_estimators = 300
    st.number_input(
        "Random Forest estimators",
        value=rf_estimators,
        disabled=True
    )

destinos_df[["latitud", "longitud"]] = destinos_df["coordenadas_gps"].str.split(",", expand=True)
destinos_df["latitud"] = pd.to_numeric(destinos_df["latitud"], errors="coerce")
destinos_df["longitud"] = pd.to_numeric(destinos_df["longitud"], errors="coerce")

cantidades = lineas_df.groupby("PedidoID")["Cantidad"].sum().reset_index(name="Cantidad_total")

lineas_productos = lineas_df.merge(productos_df, on="ProductoID", how="left")
lineas_productos["TiempoProducto"] = lineas_productos["TiempoFabricacionMedia"] + lineas_productos["Caducidad"]
tiempo_min = lineas_productos.groupby("PedidoID")["TiempoProducto"].min().reset_index(name="TiempoTotalEstimado")

pedidos_destino = pedidos_df.merge(destinos_df, left_on="DestinoEntregaID", right_on="DestinoID", how="left")
pedidos_final = pedidos_destino.merge(cantidades, on="PedidoID", how="left").merge(tiempo_min, on="PedidoID", how="left")
pedidos_final = pedidos_final.sort_values("PedidoID").reset_index(drop=True)


if "ciudad_y" in pedidos_final.columns:
    pedidos_final.rename(columns={"ciudad_y": "ciudad"}, inplace=True)
if "ciudad_x" in pedidos_final.columns:
    pedidos_final.drop(columns=["ciudad_x"], inplace=True)

before_count = len(pedidos_final)
pedidos_final["ciudad"] = pedidos_final["ciudad"].astype(object)
pedidos_final["ciudad"] = pedidos_final["ciudad"].apply(lambda x: x.strip() if isinstance(x, str) else x)
pedidos_final = pedidos_final[pedidos_final["ciudad"].notna() & (pedidos_final["ciudad"] != "")]
removed = before_count - len(pedidos_final)
if removed > 0:
    st.warning(f"Se han eliminado {removed} pedidos sin 'ciudad' asociada.")

pedidos_final["latitud"] = pd.to_numeric(pedidos_final.get("latitud"), errors="coerce")
pedidos_final["longitud"] = pd.to_numeric(pedidos_final.get("longitud"), errors="coerce")
coords_valid = pedidos_final.dropna(subset=["latitud", "longitud"]).copy()

st.header("3) Zonificación geográfica (cluster)")
if coords_valid.empty:
    st.warning("No hay coordenadas válidas; todos los pedidos tendrán ClusterID = -1.")
    pedidos_final["ClusterID"] = -1
    pedidos_final["DistanciaCluster_km"] = 9999.0
else:
    n_clusters_eff = min(n_clusters, max(1, len(coords_valid)))
    kmeans = KMeans(n_clusters=n_clusters_eff, random_state=42, n_init=10)
    kmeans_fit = kmeans.fit(coords_valid[["latitud", "longitud"]].values)
    coords_valid = coords_valid.assign(ClusterID=kmeans_fit.labels_)
    centers = kmeans_fit.cluster_centers_
    def dist_to_center(row):
        lab = int(row["ClusterID"])
        center_lat, center_lon = centers[lab]
        return haversine(row["latitud"], row["longitud"], center_lat, center_lon)
    coords_valid["DistanciaCluster_km"] = coords_valid.apply(dist_to_center, axis=1)
    pedidos_final = pedidos_final.merge(coords_valid[["PedidoID", "ClusterID", "DistanciaCluster_km"]], on="PedidoID", how="left")
    pedidos_final["ClusterID"].fillna(-1, inplace=True)
    pedidos_final["DistanciaCluster_km"].fillna(9999.0, inplace=True)

st.write("Preview pedidos (con cluster):")
preview_cols = ["PedidoID", "ciudad", "latitud", "longitud", "Cantidad_total", "TiempoTotalEstimado", "ClusterID", "DistanciaCluster_km"]
st.dataframe(pedidos_final[[c for c in preview_cols if c in pedidos_final.columns]].head(10))

st.markdown("---")
st.header("4) Entrenar Random Forest y asignar pedidos (prioridad: minimizar km y por tiempo)")

if st.button("Entrenar y asignar con Random Forest (prioridad: minimizar km + tiempo)"):
    st.info("Construyendo dataset sintético (etiquetas) y entrenando Random Forest...")

    pf = pedidos_final.copy()
    if pf["TiempoTotalEstimado"].dropna().empty:
        pf["Tiempo_filled"] = 1.0
    else:
        max_time_existing = pf["TiempoTotalEstimado"].dropna().max()
        pf["Tiempo_filled"] = pf["TiempoTotalEstimado"].apply(lambda x: x if pd.notna(x) and x >= 0 else (max_time_existing + 1.0))
    max_time = pf["Tiempo_filled"].max() if not pf["Tiempo_filled"].empty else 1.0
    pf["Priority"] = pf["Tiempo_filled"].apply(lambda t: float(max_time - t) / float(max_time) if max_time > 0 else 0.0)
    pedidos_final = pedidos_final.merge(pf[["PedidoID", "Tiempo_filled", "Priority"]], on="PedidoID", how="left")

    samples = []
    labels = []

    orden_sintetico = pedidos_final.copy()
    orden_sintetico["Tiempo_filled"] = orden_sintetico["Tiempo_filled"].fillna(orden_sintetico["Tiempo_filled"].max() if not orden_sintetico["Tiempo_filled"].dropna().empty else 1.0)
    orden_sintetico = orden_sintetico.sort_values(["Tiempo_filled", "ClusterID", "DistanciaCluster_km"]).reset_index(drop=True)
    
    trucks_state = [{"id":1, "cap": capacidad_camion, "orders": [], "clusters": set(), "last_lat": ORIGEN[0], "last_lon": ORIGEN[1], "tiempos": set(), "priorities": []}]
    for _, row in orden_sintetico.iterrows():
        pedido_id = row["PedidoID"]
        qty = float(row["Cantidad_total"] if pd.notna(row["Cantidad_total"]) else 0)
        tiempo = float(row["TiempoTotalEstimado"] if pd.notna(row["TiempoTotalEstimado"]) else -1)
        cluster = int(row.get("ClusterID", -1))
        city = row.get("ciudad", "")
        lat = row.get("latitud"); lon = row.get("longitud")
        dist_origin = 9999.0 if pd.isna(lat) or pd.isna(lon) else haversine(ORIGEN[0], ORIGEN[1], lat, lon)
        priority = float(row.get("Priority", 0.0))

        # 1) preferir camiones que ya tengan el mismo tiempo estimado
        # 2) preferir camiones que tengan pedidos con prioridad similar
        # 3) preferir camiones que tengan el mismo cluster
        # 4) luego cualquier camión con capacidad
        assigned_idx = None
        for idx, t in enumerate(trucks_state):
            if t["cap"] >= qty and (tiempo in t["tiempos"]):
                assigned_idx = idx
                break
        if assigned_idx is None:
            for idx, t in enumerate(trucks_state):
                if t["cap"] >= qty and t["priorities"]:
                    if any(abs(p - priority) <= 0.05 for p in t["priorities"]):
                        assigned_idx = idx
                        break
        if assigned_idx is None:
            for idx, t in enumerate(trucks_state):
                if t["cap"] >= qty and (cluster in t["clusters"] and cluster != -1):
                    assigned_idx = idx
                    break
        if assigned_idx is None:
            for idx, t in enumerate(trucks_state):
                if t["cap"] >= qty:
                    assigned_idx = idx
                    break
        if assigned_idx is None:
            trucks_state.append({"id": len(trucks_state)+1, "cap": capacidad_camion, "orders": [], "clusters": set(), "last_lat": ORIGEN[0], "last_lon": ORIGEN[1], "tiempos": set(), "priorities": []})
            assigned_idx = len(trucks_state)-1

        cap_before = trucks_state[assigned_idx]["cap"]
        last_lat = trucks_state[assigned_idx]["last_lat"]
        last_lon = trucks_state[assigned_idx]["last_lon"]
        same_city_flag = int(any(o.get("ciudad","") == city for o in trucks_state[assigned_idx]["orders"]))
        same_tiempo_flag = int(tiempo in trucks_state[assigned_idx]["tiempos"])
        marginal_km = estimate_marginal_km(last_lat, last_lon, lat, lon)

        samples.append({
            "Cantidad_total": qty,
            "TiempoTotalEstimado": tiempo,
            "Distancia_origen_km": dist_origin,
            "Capacidad_restante": cap_before,
            "Capacidad_ratio": qty / capacidad_camion if capacidad_camion>0 else 0,
            "ClusterID": cluster,
            "DistanciaCluster_km": float(row.get("DistanciaCluster_km", 9999.0)),
            "SameCity": same_city_flag,
            "SameTiempo": same_tiempo_flag,
            "Marginal_km": marginal_km,
            "Priority": priority
        })
        labels.append(1)

        other_idxs = [i for i in range(len(trucks_state)) if i != assigned_idx]
        neg_candidates = other_idxs[:2]
        if not neg_candidates:
            neg_candidates = [None]

        for oi in neg_candidates:
            if oi is None:
                last_lat_neg, last_lon_neg = ORIGEN[0], ORIGEN[1]
                cap_before_neg = capacidad_camion
                same_city_neg = 0
                same_tiempo_neg = 0
                priority_neg = 0.0
            else:
                last_lat_neg = trucks_state[oi]["last_lat"]
                last_lon_neg = trucks_state[oi]["last_lon"]
                cap_before_neg = trucks_state[oi]["cap"]
                same_city_neg = int(any(o.get("ciudad","") == city for o in trucks_state[oi]["orders"]))
                same_tiempo_neg = int(tiempo in trucks_state[oi]["tiempos"])
                priority_neg = 1.0 if any(abs(p - priority) <= 0.05 for p in trucks_state[oi]["priorities"]) else 0.0

            marginal_neg = estimate_marginal_km(last_lat_neg, last_lon_neg, lat, lon)

            samples.append({
                "Cantidad_total": qty,
                "TiempoTotalEstimado": tiempo,
                "Distancia_origen_km": dist_origin,
                "Capacidad_restante": cap_before_neg,
                "Capacidad_ratio": qty / capacidad_camion if capacidad_camion>0 else 0,
                "ClusterID": cluster,
                "DistanciaCluster_km": float(row.get("DistanciaCluster_km", 9999.0)),
                "SameCity": same_city_neg,
                "SameTiempo": same_tiempo_neg,
                "Marginal_km": marginal_neg,
                "Priority": priority_neg
            })
            labels.append(0)

        trucks_state[assigned_idx]["cap"] -= qty
        trucks_state[assigned_idx]["orders"].append({"PedidoID": pedido_id, "ciudad": city, "lat": lat, "lon": lon, "TiempoTotalEstimado": tiempo})
        trucks_state[assigned_idx]["clusters"].add(cluster)
        trucks_state[assigned_idx]["tiempos"].add(tiempo)
        trucks_state[assigned_idx]["priorities"].append(priority)
        if pd.notna(lat) and pd.notna(lon):
            trucks_state[assigned_idx]["last_lat"] = lat
            trucks_state[assigned_idx]["last_lon"] = lon

    X = pd.DataFrame(samples)
    y = pd.Series(labels)
    X["ClusterID"] = X["ClusterID"].fillna(-1).astype(int)
    X["DistanciaCluster_km"] = pd.to_numeric(X["DistanciaCluster_km"], errors="coerce").fillna(9999.0)
    X["SameCity"] = X["SameCity"].fillna(0).astype(int)
    X["SameTiempo"] = X["SameTiempo"].fillna(0).astype(int)
    X["Marginal_km"] = pd.to_numeric(X["Marginal_km"], errors="coerce").fillna(9999.0)
    X["Priority"] = pd.to_numeric(X["Priority"], errors="coerce").fillna(0.0)

    feat_cols = ["Cantidad_total", "TiempoTotalEstimado", "Distancia_origen_km",
                 "Capacidad_restante", "Capacidad_ratio",
                 "ClusterID", "DistanciaCluster_km",
                 "SameCity", "SameTiempo", "Marginal_km", "Priority"]
    X = X[feat_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())>1 else None)
    rf = RandomForestClassifier(n_estimators=int(rf_estimators), random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"Modelo entrenado. Accuracy (test): {acc:.3f}")
    st.session_state["rf_model"] = rf
    st.session_state["rf_features"] = feat_cols

    st.info("Aplicando modelo (RF + heurísticas km-aware + prioridad por tiempo + mejora local)...")
    model_trucks = [{"id": 1, "cap_remaining": capacidad_camion, "orders": [], "clusters": set(), "last_lat": ORIGEN[0], "last_lon": ORIGEN[1], "tiempos": set(), "priorities": []}]

    all_marginals = []
    for _, row in pedidos_final.iterrows():
        lat = row.get("latitud"); lon = row.get("longitud")
        if pd.notna(lat) and pd.notna(lon):
            all_marginals.append(haversine(ORIGEN[0], ORIGEN[1], lat, lon))
    max_marginal = max(all_marginals) if all_marginals else 1.0
    max_marginal = max(max_marginal, 1.0)

    pf2 = pedidos_final.copy()
    pf2["Tiempo_filled"] = pf2["Tiempo_filled"].fillna(pf2["Tiempo_filled"].max() if not pf2["Tiempo_filled"].dropna().empty else 1.0)
    pf2 = pf2.sort_values(["Tiempo_filled", "ClusterID", "DistanciaCluster_km"]).reset_index(drop=True)

    for _, row in pf2.iterrows():
        pedido_id = row["PedidoID"]
        qty = float(row["Cantidad_total"] if pd.notna(row["Cantidad_total"]) else 0)
        tiempo = float(row["TiempoTotalEstimado"] if pd.notna(row["TiempoTotalEstimado"]) else -1)
        cluster = int(row.get("ClusterID", -1))
        city = row.get("ciudad", "")
        lat = row.get("latitud"); lon = row.get("longitud")
        dist_origin = 9999.0 if pd.isna(lat) or pd.isna(lon) else haversine(ORIGEN[0], ORIGEN[1], lat, lon)
        priority = float(row.get("Priority", 0.0))

        candidates = []
        for t in model_trucks:
            candidates.append({"truck_id": t["id"], "cap_before": t["cap_remaining"], "last_lat": t["last_lat"], "last_lon": t["last_lon"], "orders": t["orders"], "tiempos": t["tiempos"], "priorities": t.get("priorities", [])})
        candidates.append({"truck_id": len(model_trucks)+1, "cap_before": capacidad_camion, "last_lat": ORIGEN[0], "last_lon": ORIGEN[1], "orders": [], "tiempos": set(), "priorities": []})

        feats = []
        valid_candidate_indices = []
        for idx, c in enumerate(candidates):
            if c["cap_before"] >= qty:
                samecity = int(any(o.get("ciudad","") == city for o in c["orders"]))
                sametiempo = int(tiempo in c.get("tiempos", set()))
                samepriority = int(any(abs(p - priority) <= 0.05 for p in c.get("priorities", [])))
                marginal = estimate_marginal_km(c["last_lat"], c["last_lon"], lat, lon)
                feats.append({
                    "Cantidad_total": qty,
                    "TiempoTotalEstimado": tiempo,
                    "Distancia_origen_km": dist_origin,
                    "Capacidad_restante": c["cap_before"],
                    "Capacidad_ratio": qty / capacidad_camion if capacidad_camion>0 else 0,
                    "ClusterID": cluster,
                    "DistanciaCluster_km": float(row.get("DistanciaCluster_km", 9999.0)),
                    "SameCity": samecity,
                    "SameTiempo": sametiempo,
                    "Marginal_km": marginal,
                    "Priority": priority
                })
                valid_candidate_indices.append(idx)
        if not feats:
            new_id = len(model_trucks) + 1
            model_trucks.append({"id": new_id, "cap_remaining": capacidad_camion - qty, "orders": [{"PedidoID": row["PedidoID"], "ciudad": city, "lat": lat, "lon": lon, "TiempoTotalEstimado": tiempo}], "clusters": {cluster}, "last_lat": lat if pd.notna(lat) else ORIGEN[0], "last_lon": lon if pd.notna(lon) else ORIGEN[1], "tiempos": {tiempo}, "priorities": [priority]})
            continue

        Xc = pd.DataFrame(feats)[feat_cols]
        probs = rf.predict_proba(Xc)[:, 1]

        w_marg = 0.25
        w_same = 0.25
        w_priority = 0.30
        scores = []
        for i, p in enumerate(probs):
            feat = feats[i]
            norm_marg = feat["Marginal_km"] / max_marginal
            score = float(p) - w_marg * float(norm_marg) + w_same * float(feat.get("SameTiempo", 0)) + w_priority * float(feat.get("Priority", 0.0))
            scores.append(score)

        best_idx_local = int(np.argmax(scores))
        chosen_global_idx = valid_candidate_indices[best_idx_local]
        chosen = candidates[chosen_global_idx]
        chosen_truck_id = chosen["truck_id"]

        if chosen_truck_id == len(model_trucks)+1:
            model_trucks.append({"id": chosen_truck_id, "cap_remaining": capacidad_camion - qty, "orders": [{"PedidoID": row["PedidoID"], "ciudad": city, "lat": lat, "lon": lon, "TiempoTotalEstimado": tiempo}], "clusters": {cluster}, "last_lat": lat if pd.notna(lat) else ORIGEN[0], "last_lon": lon if pd.notna(lon) else ORIGEN[1], "tiempos": {tiempo}, "priorities": [priority]})
        else:
            for t in model_trucks:
                if t["id"] == chosen_truck_id:
                    t["orders"].append({"PedidoID": row["PedidoID"], "ciudad": city, "lat": lat, "lon": lon, "TiempoTotalEstimado": tiempo})
                    t["cap_remaining"] -= qty
                    t["clusters"].add(cluster)
                    t.setdefault("tiempos", set()).add(tiempo)
                    t.setdefault("priorities", []).append(priority)
                    if pd.notna(lat) and pd.notna(lon):
                        t["last_lat"] = lat
                        t["last_lon"] = lon
                    break
                
    st.info("Aplicando mejora local (movimientos/swap + 2-opt por ruta) para reducir km totales...")

    def compute_total_distance_all(trucks_list):
        total = 0.0
        for tt in trucks_list:
            stops = [{"lat": o["lat"], "lon": o["lon"]} for o in tt["orders"] if pd.notna(o.get("lat")) and pd.notna(o.get("lon"))]
            total += route_distance_for_stops(stops)
        return total

    for t in model_trucks:
        if t["orders"]:
            stops = [o for o in t["orders"] if pd.notna(o.get("lat")) and pd.notna(o.get("lon"))]
            if stops:
                improved_order = two_opt_route_order(ORIGEN, stops, max_iter=100)
                map_by_id = {o["PedidoID"]: o for o in t["orders"]}
                new_orders = []
                for s in improved_order:
                    pid = s.get("PedidoID")
                    if pid in map_by_id:
                        new_orders.append(map_by_id[pid])
                    else:
                        new_orders.append({"PedidoID": s.get("pedido", None), "ciudad": s.get("city", None), "lat": s["lat"], "lon": s["lon"], "TiempoTotalEstimado": s.get("TiempoTotalEstimado", -1)})
                t["orders"] = new_orders
                if new_orders:
                    last = new_orders[-1]
                    t["last_lat"] = last["lat"] if pd.notna(last.get("lat")) else ORIGEN[0]
                    t["last_lon"] = last["lon"] if pd.notna(last.get("lon")) else ORIGEN[1]

    improved = True
    max_iters = 50
    iter_count = 0
    while improved and iter_count < max_iters:
        iter_count += 1
        improved = False
        base_total = compute_total_distance_all(model_trucks)

        for t_idx, t in enumerate(model_trucks):
            for order in t["orders"][:]:
                pid = order["PedidoID"]
                qty = float(order.get("Cantidad_total", 0)) if "Cantidad_total" in order else float(pedidos_final.loc[pedidos_final["PedidoID"]==pid, "Cantidad_total"].iloc[0]) if any(pedidos_final["PedidoID"]==pid) else 0
                for target_idx in range(len(model_trucks)+1):
                    if target_idx == t_idx:
                        continue
                    tmp_trucks = copy.deepcopy(model_trucks)
                    removed = False
                    for o in tmp_trucks[t_idx]["orders"]:
                        if o["PedidoID"] == pid:
                            tmp_trucks[t_idx]["orders"].remove(o)
                            tmp_trucks[t_idx]["cap_remaining"] += qty
                            removed = True
                            break
                    if not removed:
                        continue
                    if target_idx == len(model_trucks):
                        new_truck = {"id": len(tmp_trucks)+1, "cap_remaining": capacidad_camion - qty, "orders": [order], "clusters": {int(order.get("ClusterID", -1))}, "last_lat": order.get("lat", ORIGEN[0]), "last_lon": order.get("lon", ORIGEN[1]), "tiempos": {order.get("TiempoTotalEstimado", -1)}, "priorities": [float(pedidos_final.loc[pedidos_final['PedidoID']==pid, 'Priority'].iloc[0]) if any(pedidos_final['PedidoID']==pid) else 0.0]}
                        tmp_trucks.append(new_truck)
                    else:
                        if tmp_trucks[target_idx]["cap_remaining"] < qty:
                            continue
                        tmp_trucks[target_idx]["orders"].append(order)
                        tmp_trucks[target_idx]["cap_remaining"] -= qty
                        tmp_trucks[target_idx].setdefault("tiempos", set()).add(order.get("TiempoTotalEstimado", -1))
                        tmp_trucks[target_idx].setdefault("priorities", []).append(float(pedidos_final.loc[pedidos_final['PedidoID']==pid, 'Priority'].iloc[0]) if any(pedidos_final['PedidoID']==pid) else 0.0)
                    affected_indices = [t_idx]
                    if target_idx < len(model_trucks):
                        affected_indices.append(target_idx)
                    else:
                        affected_indices.append(len(tmp_trucks)-1)
                    for affected in affected_indices:
                        if affected < 0 or affected >= len(tmp_trucks):
                            continue
                        stops_tmp = [o for o in tmp_trucks[affected]["orders"] if pd.notna(o.get("lat")) and pd.notna(o.get("lon"))]
                        if stops_tmp:
                            new_order_tmp = two_opt_route_order(ORIGEN, stops_tmp, max_iter=80)
                            map_by_id_tmp = {o["PedidoID"]: o for o in tmp_trucks[affected]["orders"]}
                            new_orders2 = []
                            for s in new_order_tmp:
                                pid2 = s.get("PedidoID")
                                if pid2 in map_by_id_tmp:
                                    new_orders2.append(map_by_id_tmp[pid2])
                                else:
                                    new_orders2.append({"PedidoID": s.get("pedido"), "ciudad": s.get("city"), "lat": s["lat"], "lon": s["lon"], "TiempoTotalEstimado": s.get("TiempoTotalEstimado", -1)})
                            tmp_trucks[affected]["orders"] = new_orders2
                            if new_orders2:
                                last = new_orders2[-1]
                                tmp_trucks[affected]["last_lat"] = last["lat"] if pd.notna(last.get("lat")) else ORIGEN[0]
                                tmp_trucks[affected]["last_lon"] = last["lon"] if pd.notna(last.get("lon")) else ORIGEN[1]

                    new_total = compute_total_distance_all(tmp_trucks)
                    if new_total + 1e-6 < base_total:
                        model_trucks = tmp_trucks
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            for (i, j) in itertools.combinations(range(len(model_trucks)), 2):
                truck_i = model_trucks[i]
                truck_j = model_trucks[j]
                for oi in truck_i["orders"][:]:
                    for oj in truck_j["orders"][:]:
                        pid_i = oi["PedidoID"]; pid_j = oj["PedidoID"]
                        qty_i = float(oi.get("Cantidad_total", 0)) if "Cantidad_total" in oi else float(pedidos_final.loc[pedidos_final["PedidoID"]==pid_i, "Cantidad_total"].iloc[0]) if any(pedidos_final["PedidoID"]==pid_i) else 0
                        qty_j = float(oj.get("Cantidad_total", 0)) if "Cantidad_total" in oj else float(pedidos_final.loc[pedidos_final["PedidoID"]==pid_j, "Cantidad_total"].iloc[0]) if any(pedidos_final["PedidoID"]==pid_j) else 0
                        if truck_i["cap_remaining"] + qty_i - qty_j < 0 or truck_j["cap_remaining"] + qty_j - qty_i < 0:
                            continue
                        tmp_trucks = copy.deepcopy(model_trucks)
                        ti = tmp_trucks[i]; tj = tmp_trucks[j]
                        ti["orders"] = [o for o in ti["orders"] if o["PedidoID"] != pid_i]
                        tj["orders"] = [o for o in tj["orders"] if o["PedidoID"] != pid_j]
                        ti["orders"].append(oj)
                        tj["orders"].append(oi)
                        ti["cap_remaining"] = ti["cap_remaining"] + qty_i - qty_j
                        tj["cap_remaining"] = tj["cap_remaining"] + qty_j - qty_i
                        for affected in [i, j]:
                            stops_tmp = [o for o in tmp_trucks[affected]["orders"] if pd.notna(o.get("lat")) and pd.notna(o.get("lon"))]
                            if stops_tmp:
                                new_order_tmp = two_opt_route_order(ORIGEN, stops_tmp, max_iter=80)
                                map_by_id_tmp = {o["PedidoID"]: o for o in tmp_trucks[affected]["orders"]}
                                new_orders2 = []
                                for s in new_order_tmp:
                                    pid2 = s.get("PedidoID")
                                    if pid2 in map_by_id_tmp:
                                        new_orders2.append(map_by_id_tmp[pid2])
                                    else:
                                        new_orders2.append({"PedidoID": s.get("pedido"), "ciudad": s.get("city"), "lat": s["lat"], "lon": s["lon"], "TiempoTotalEstimado": s.get("TiempoTotalEstimado", -1)})
                                tmp_trucks[affected]["orders"] = new_orders2
                                if new_orders2:
                                    last = new_orders2[-1]
                                    tmp_trucks[affected]["last_lat"] = last["lat"] if pd.notna(last.get("lat")) else ORIGEN[0]
                                    tmp_trucks[affected]["last_lon"] = last["lon"] if pd.notna(last.get("lon")) else ORIGEN[1]
                        new_total = compute_total_distance_all(tmp_trucks)
                        base_total2 = compute_total_distance_all(model_trucks)
                        if new_total + 1e-6 < base_total2:
                            model_trucks = tmp_trucks
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    # construir lista de camiones
    model_camiones = []
    for t in model_trucks:
        if t["orders"]:
            df_orders = pd.DataFrame(t["orders"])
            if "PedidoID" in df_orders.columns:
                df_full = pd.merge(df_orders, pedidos_final, on="PedidoID", how="left", suffixes=("_orders", "_orig"))
            else:
                df_full = df_orders.copy()
            if "ciudad" not in df_full.columns:
                if "ciudad_orders" in df_full.columns:
                    df_full.rename(columns={"ciudad_orders": "ciudad"}, inplace=True)
                elif "ciudad_orig" in df_full.columns:
                    df_full.rename(columns={"ciudad_orig": "ciudad"}, inplace=True)
                else:
                    possible_city_cols = [c for c in df_full.columns if c.lower().startswith("ciudad") or c.lower().startswith("city")]
                    if possible_city_cols:
                        df_full["ciudad"] = df_full[possible_city_cols[0]]
                    else:
                        df_full["ciudad"] = None

            for col in ["Cantidad_total", "TiempoTotalEstimado", "ClusterID", "latitud", "longitud"]:
                if col not in df_full.columns:
                    if col + "_orig" in df_full.columns:
                        df_full[col] = df_full[col + "_orig"]
                    else:
                        df_full[col] = pd.NA

            model_camiones.append((t["id"], df_full.reset_index(drop=True)))

    st.session_state["model_camiones"] = model_camiones
    st.session_state["model_trucks_raw"] = model_trucks

    for t in model_trucks:
        stops = []
        for o in t.get("orders", []):
            lat = o.get("lat"); lon = o.get("lon")
            if pd.notna(lat) and pd.notna(lon):
                stops.append({"PedidoID": o.get("PedidoID"), "lat": float(lat), "lon": float(lon)})
        if not stops:
            t["total_km"] = 0.0
            t["km_to_last"] = 0.0
            t["cost_eur"] = 0.0
            t["n_pedidos"] = 0
            continue

        ordered = two_opt_route_order(ORIGEN, stops, max_iter=100)

        coords = [ORIGEN_LONLAT]
        for s in ordered:
            coords.append([s["lon"], s["lat"]])
        coords.append(ORIGEN_LONLAT)

        total_km = 0.0
        km_to_last = 0.0
        total_legs = len(coords) - 1
        for i in range(total_legs):
            a, b = coords[i], coords[i+1]
            try:
                if client is None:
                    raise RuntimeError("ORS client no disponible")
                resp = client.directions(coordinates=[a,b], profile="driving-car", format="json", radiuses=[1000,1000])
                route = resp["routes"][0]
                leg_km = route["summary"]["distance"]/1000.0
                total_km += leg_km
                if i != total_legs - 1:
                    km_to_last += leg_km
            except Exception:
                latlon_a = [a[1], a[0]]
                latlon_b = [b[1], b[0]]
                leg_km = haversine(latlon_a[0], latlon_a[1], latlon_b[0], latlon_b[1])
                total_km += leg_km
                if i != total_legs - 1:
                    km_to_last += leg_km

        cost_eur = (total_km / 100.0) * consumo_l_100km * precio_litro
        t["total_km"] = total_km
        t["km_to_last"] = km_to_last
        t["cost_eur"] = cost_eur
        t["n_pedidos"] = len(stops)

    st.session_state["model_trucks_raw"] = model_trucks

    sum_total_km = sum(t.get("total_km", 0.0) for t in model_trucks)
    total_cost_all = sum(t.get("cost_eur", 0.0) for t in model_trucks)

    st.success(f"Asignación completada — camiones usados: {len(model_camiones)}")
    st.success(f"Km total (todos los camiones, origen→origen) [suma de rutas por camión]: {sum_total_km:.2f} km")
    st.success(f"Coste total combustible (todos los camiones): {total_cost_all:.2f} €")

st.markdown("---")
st.header("5) Resultado de asignación (RF)")

if "model_camiones" not in st.session_state:
    st.info("No hay asignación realizada aún. Pulsa 'Entrenar y asignar con Random Forest (prioridad: minimizar km + tiempo)'.")
else:
    for tid, df_tr in st.session_state["model_camiones"]:
        with st.expander(f"Camión {tid} — pedidos: {len(df_tr)} — carga total: {pd.to_numeric(df_tr.get('Cantidad_total', pd.Series([0]))).sum():.0f}", expanded=False):
            desired_cols = ["PedidoID", "ciudad", "Cantidad_total", "TiempoTotalEstimado", "ClusterID"]
            cols_show = [c for c in desired_cols if c in df_tr.columns]
            if not cols_show:
                st.dataframe(df_tr.head(50))
            else:
                st.dataframe(df_tr[cols_show].reset_index(drop=True))

st.markdown("---")
st.header("6) Mostrar ruta por camión (visualización)")

def mostrar_mapa_camion_from_df(df_camion, camion_id):
    stops = []
    for _, row in df_camion.iterrows():
        # try original gps string first
        gps_field = None
        if "coordenadas_gps" in row and pd.notna(row["coordenadas_gps"]):
            gps_field = row["coordenadas_gps"]
        elif "coordenadas_gps_orders" in row and pd.notna(row["coordenadas_gps_orders"]):
            gps_field = row["coordenadas_gps_orders"]
        if gps_field is not None:
            parsed = parse_latlon(gps_field)
        else:
            parsed = None

        if not parsed:
            lat = row.get("latitud") if "latitud" in row else row.get("lat")
            lon = row.get("longitud") if "longitud" in row else row.get("lon")
            if pd.isna(lat) or pd.isna(lon):
                st.warning(f"Pedido {row.get('PedidoID')} descartado (coords inválidas)")
                continue
            parsed = (float(lat), float(lon))
        lat, lon = parsed
        stops.append({"PedidoID": row.get("PedidoID"), "city": row.get("ciudad"), "lat": lat, "lon": lon})

    if not stops:
        st.error("No hay paradas válidas para este camión.")
        return

    ordered_stops = two_opt_route_order(ORIGEN, stops, max_iter=200)

    coords = [ORIGEN_LONLAT]
    final_stops = [{"city": "Origen", "lat": ORIGEN[0], "lon": ORIGEN[1]}]
    for s in ordered_stops:
        coords.append([s["lon"], s["lat"]])
        final_stops.append(s)
    coords.append(ORIGEN_LONLAT)
    final_stops.append({"city": "Origen", "lat": ORIGEN[0], "lon": ORIGEN[1]})

    m = folium.Map(location=[final_stops[1]["lat"], final_stops[1]["lon"]], zoom_start=8)
    cluster = MarkerCluster().add_to(m)
    folium.Marker(ORIGEN, popup="Origen", icon=folium.Icon(color="blue", icon="home")).add_to(m)
    for i, s in enumerate(final_stops[1:-1], start=1):
        folium.Marker([s["lat"], s["lon"]], popup=f"Pedido {s['PedidoID']} - {s['city']}", tooltip=s['city']).add_to(cluster)

    total_km_local = 0.0
    km_to_last_local = 0.0
    total_legs = len(coords) - 1
    for i in range(total_legs):
        a, b = coords[i], coords[i+1]
        try:
            if client is None:
                raise RuntimeError("ORS client no disponible")
            resp = client.directions(coordinates=[a,b], profile="driving-car", format="json", radiuses=[1000,1000])
            route = resp["routes"][0]
            leg_km = route["summary"]["distance"]/1000
            total_km_local += leg_km
            if i != total_legs - 1:
                km_to_last_local += leg_km
            decoded = convert.decode_polyline(route["geometry"])["coordinates"]
            poly = [(lat, lon) for lon, lat in decoded]
            if i == total_legs - 1:
                folium.PolyLine(poly, color="black", weight=4, dash_array="6,8", tooltip="Retorno al origen").add_to(m)
            else:
                color = hsv_to_hex(i/max(1,total_legs-2))
                folium.PolyLine(poly, color=color, weight=5, tooltip=f"{final_stops[i]['city']} → {final_stops[i+1]['city']}").add_to(m)
        except Exception:
            latlon_a = [a[1], a[0]]
            latlon_b = [b[1], b[0]]
            folium.PolyLine([latlon_a, latlon_b], color="gray", weight=3, dash_array="4,6").add_to(m)
            leg_km = haversine(latlon_a[0], latlon_a[1], latlon_b[0], latlon_b[1])
            total_km_local += leg_km
            if i != total_legs - 1:
                km_to_last_local += leg_km

    legend_html = """
    <div style="
        position: fixed; bottom: 40px; left: 40px; z-index: 9999;
        background: white; padding: 10px; border: 2px solid grey; font-size: 14px;
    ">
      <b>Leyenda</b><br>
      <span style="color:red;">━━━</span> Tramos de entrega<br>
      <span style="color:black;">- - -</span> Retorno al origen
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    st.components.v1.html(m._repr_html_(), height=700)

    stored = None
    if "model_trucks_raw" in st.session_state:
        stored = next((x for x in st.session_state["model_trucks_raw"] if x.get("id") == camion_id), None)

    if stored is not None and "total_km" in stored:
        total_km_show = stored["total_km"]
        km_to_last_show = stored.get("km_to_last", km_to_last_local)
    else:
        total_km_show = total_km_local
        km_to_last_show = km_to_last_local

    coste_combustible_total = (total_km_show / 100.0) * consumo_l_100km * precio_litro

    st.success(f"Distancia total aproximada (euclídea): {total_km_show:.2f} km")
    st.success(f"Coste total de combustible del trayecto completo (incluye retorno al origen): {coste_combustible_total:.2f} €")
    st.info(f"Distancia hasta el último pedido (sin retorno al origen): {km_to_last_show:.2f} km")

    try:
        vm = float(velocidad_media)
        if vm <= 0:
            st.warning("Velocidad media debe ser mayor que 0 para calcular tiempos.")
        else:
            hours_to_last = km_to_last_show / vm
            driving_hours_per_day = 9.0
            days_to_last = hours_to_last / driving_hours_per_day
            st.info(f"Tiempo de conducción estimado hasta el último pedido: {hours_to_last:.2f} h")
            st.info(f"Días de conducción estimados hasta el último pedido (conductor: {driving_hours_per_day:.0f} h/día): {days_to_last:.2f} días")

            hours_total = total_km_show / vm
            days_total = hours_total / driving_hours_per_day

            st.info(f"Tiempo de conducción estimado hasta volver al origen (Mataró): {hours_total:.2f} h")
            st.info(f"Días de conducción estimados hasta volver al origen (conductor: {driving_hours_per_day:.0f} h/día): {days_total:.2f} días")
    except Exception:
        st.warning("No se pudo calcular tiempo estimado hasta el último pedido (error en velocidad media).")

# botones por camión
if "model_camiones" in st.session_state:
    for camion_id, df_camion in st.session_state["model_camiones"]:
        st.subheader(f"🚚 Camión {camion_id}")
        total_qty = pd.to_numeric(df_camion.get('Cantidad_total', pd.Series([0])), errors="coerce").fillna(0).sum()
        st.write(f"Pedidos: {len(df_camion)} — Carga total: {total_qty:.0f}")
        if st.button(f"Mostrar ruta Camión {camion_id}", key=f"show_{camion_id}"):
            mostrar_mapa_camion_from_df(df_camion, camion_id)
else:
    st.info("No hay camiones asignados para visualizar. Ejecuta la asignación primero.")

if "model_trucks_raw" in st.session_state:
    trucks_all = st.session_state["model_trucks_raw"]

    per_truck = []
    sum_total_km = 0.0
    sum_cost = 0.0
    for t in trucks_all:
        km_t = t.get("total_km", 0.0)
        cost_t = t.get("cost_eur", 0.0)
        n_ped = t.get("n_pedidos", 0)
        per_truck.append({"truck_id": t.get("id"), "km": km_t, "cost_eur": cost_t, "n_pedidos": n_ped})
        sum_total_km += km_t
        sum_cost += cost_t

    st.markdown("---")
    st.subheader("Resumen global: coste combustible (todos los camiones) — basado en valores mostrados arriba")
    df_cost = pd.DataFrame(per_truck)
    if not df_cost.empty:
        df_cost["km"] = df_cost["km"].round(2)
        df_cost["cost_eur"] = df_cost["cost_eur"].round(2)
        st.dataframe(df_cost.rename(columns={"cost_eur": "cost (€)", "n_pedidos": "n pedidos"}).reset_index(drop=True))
    else:
        st.info("No hay camiones con paradas válidas para calcular coste.")

    st.success(f"Km total (todos los camiones, origen→origen): {sum_total_km:.2f} km")
    st.success(f"Coste total combustible (todos los camiones): {sum_cost:.2f} €")