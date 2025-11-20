# core/graph.py
# -*- coding: utf-8 -*-
# Construcción de grafo completo con distancias geodésicas

import networkx as nx
from geopy.distance import geodesic

ORIGEN_ID = "Mataro"
ORIGEN_LATLON = (41.5381, 2.4445)  # coordenadas aproximadas de Mataró

def _latlon(row):
    try:
        return (float(row["lat"]), float(row["lon"]))
    except Exception:
        return None

def construir_grafo(destinos_df):
    """
    Crea un grafo no dirigido con:
      - Nodo origen (Mataró)
      - Todos los destinos con lat/lon válidos
      - Aristas completas entre todos los destinos y el origen
    Peso = distancia geodésica (km)
    """
    G = nx.Graph()
    G.add_node(ORIGEN_ID, lat=ORIGEN_LATLON[0], lon=ORIGEN_LATLON[1])

    destinos = []
    for _, r in destinos_df.iterrows():
        did = r["id_destino"]
        ll = _latlon(r)
        if not ll:
            continue
        G.add_node(did, lat=ll[0], lon=ll[1])
        G.add_edge(ORIGEN_ID, did, weight=geodesic(ORIGEN_LATLON, ll).km)
        destinos.append((did, ll))

    # Conectar destinos entre sí (grafo completo)
    for i in range(len(destinos)):
        for j in range(i + 1, len(destinos)):
            di, lli = destinos[i]
            dj, llj = destinos[j]
            G.add_edge(di, dj, weight=geodesic(lli, llj).km)
    return G
