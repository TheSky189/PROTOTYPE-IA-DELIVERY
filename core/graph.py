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