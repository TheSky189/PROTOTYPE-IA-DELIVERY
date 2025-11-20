from core.graph import construir_grafo, ORIGEN_ID
import pandas as pd

def test_grafo_basico():
    destinos = pd.DataFrame([
        ["D001","Barcelona",41.3888,2.1590],
        ["D005","Zaragoza",41.6488,-0.8891],
    ], columns=["id_destino","nombre","lat","lon"])
    G = construir_grafo(destinos)
    assert ORIGEN_ID in G
    assert "D001" in G and "D005" in G
    assert G.has_edge(ORIGEN_ID, "D001")
