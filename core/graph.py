# -*- coding: utf-8 -*-
# Construcción de grafo y distancias
import networkx as nx

def construir_grafo(destinos_df):
    """Crea un grafo simple usando dist_km_desde_mataro como proxy de distancias.
    Para demo: conectamos Mataró con cada destino y entre destinos con suma aproximada.
    """
    G = nx.Graph()
    # Nodo origen fijo
    origen = "Mataro"
    G.add_node(origen)

    # Añadir destinos
    for _, row in destinos_df.iterrows():
        d = row["id_destino"]
        G.add_node(d)
        if row.get("dist_km_desde_mataro") is not None:
            G.add_edge(origen, d, weight=float(row["dist_km_desde_mataro"]))
    return G

import folium

def generar_mapa_rutas(rutas):
    # Punto inicial centrado en Mataró
    mapa = folium.Map(location=[41.5421, 2.4445], zoom_start=7)

    for ruta in rutas:
        coords = [(p["lat"], p["lon"]) for p in ruta["paradas"]]

        # Línea de la ruta
        folium.PolyLine(coords, weight=4, opacity=0.6).add_to(mapa)

        # Marcadores por parada
        for p in ruta["paradas"]:
            folium.Marker(
                location=[p["lat"], p["lon"]],
                popup=f"{p['nombre']} (Camión {ruta['camion']})"
            ).add_to(mapa)

    return mapa
