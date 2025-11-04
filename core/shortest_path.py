# -*- coding: utf-8 -*-
# Rutinas de camino más corto
import networkx as nx

def ruta_mas_corta(G: nx.Graph, origen: str, destino: str):
    return nx.shortest_path(G, source=origen, target=destino, weight="weight")