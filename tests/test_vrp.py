import pandas as pd
from core.vrp_heuristic import planificar_rutas

def test_capacidad_no_superada():
    pedidos = pd.DataFrame([
        ["P1","2025-01-01","C1","D001",200,"2025-01-02","2025-01-05"],
        ["P2","2025-01-01","C2","D005",200,"2025-01-02","2025-01-06"],
    ], columns=["id_pedido","fecha_pedido","id_cliente","id_destino","qty_total","ready_ts","fecha_caducidad_pedido"])
    pedidos["fecha_pedido"] = pd.to_datetime(pedidos["fecha_pedido"])
    pedidos["ready_ts"] = pd.to_datetime(pedidos["ready_ts"])
    pedidos["fecha_caducidad_pedido"] = pd.to_datetime(pedidos["fecha_caducidad_pedido"])

    destinos = pd.DataFrame([
        ["D001","Barcelona",41.3888,2.1590],
        ["D005","Zaragoza",41.6488,-0.8891],
    ], columns=["id_destino","nombre","lat","lon"])

    soluciones = planificar_rutas(
        pedidos, pd.DataFrame(), destinos,
        capacidad_camion=300, vel_media=70, coste_km=0.8, max_camiones=3
    )
    s = soluciones[0]
    for r in s["rutas"]:
        assert r["carga_total"] <= 300
