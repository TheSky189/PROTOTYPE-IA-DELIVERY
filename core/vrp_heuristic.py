# core/vrp_heuristic.py
# -*- coding: utf-8 -*-
# Heurística VRP simple: capacidad + pedido indivisible + orden de visita por vecino más cercano

import pandas as pd
import networkx as nx
from .graph import construir_grafo, ORIGEN_ID

def _ruta_secuencial(G: nx.Graph, secuencia):
    """Suma de pesos (km) siguiendo la secuencia de nodos."""
    km = 0.0
    for i in range(len(secuencia) - 1):
        u, v = secuencia[i], secuencia[i + 1]
        km += G[u][v]["weight"]
    return km

def planificar_rutas(pedidos_df: pd.DataFrame, productos: pd.DataFrame, destinos_df: pd.DataFrame,
                     capacidad_camion: int, vel_media: float, coste_km: float, max_camiones: int):
    """
    Genera una única solución heurística:
      - Llenado codicioso por cercanía al origen (Mataró)
      - Visita por vecino más cercano
      - Calcula km, tiempo, coste y cumplimiento de caducidad
    """
    G = construir_grafo(destinos_df)

    def dist_origen(did):
        return G[ORIGEN_ID][did]["weight"] if did in G[ORIGEN_ID] else float("inf")

    # Ordenar pedidos por cercanía al origen (más cerca primero)
    pedidos_sorted = pedidos_df.sort_values(
        by=["id_destino"], key=lambda s: s.map(lambda d: dist_origen(d))
    ).reset_index(drop=True)

    rutas = []
    usados = set()
    vehiculo = 1

    for _ in range(max_camiones):
        carga = 0
        lote_idx = []
        destinos_lote = []

        # Llenado codicioso: añadir pedidos mientras no supere la capacidad
        for idx, row in pedidos_sorted.iterrows():
            if idx in usados:
                continue
            qty = int(row["qty_total"])
            if carga + qty <= capacidad_camion:
                carga += qty
                lote_idx.append(idx)
                destinos_lote.append(row["id_destino"])
                usados.add(idx)

        if not lote_idx:
            break

        # Secuencia por vecino más cercano
        pendientes = list(dict.fromkeys(destinos_lote))
        sec = [ORIGEN_ID]
        actual = ORIGEN_ID
        visitados = set()

        while len(visitados) < len(pendientes):
            mejor, mejor_dist = None, float("inf")
            for d in pendientes:
                if d in visitados:
                    continue
                dist = G[actual][d]["weight"]
                if dist < mejor_dist:
                    mejor, mejor_dist = d, dist
            sec.append(mejor)
            visitados.add(mejor)
            actual = mejor

        sec.append(ORIGEN_ID)

        # KPIs de la ruta
        km = _ruta_secuencial(G, sec)
        tiempo_h = km / max(vel_media, 1e-6)
        coste = km * coste_km

        # ETA por destino (acumulado)
        acumulado_km = {ORIGEN_ID: 0.0}
        total = 0.0
        for i in range(len(sec) - 1):
            u, v = sec[i], sec[i + 1]
            total += G[u][v]["weight"]
            if v not in acumulado_km:
                acumulado_km[v] = total

        cad_ok = True
        pedidos_ids = []
        for idx in lote_idx:
            row = pedidos_sorted.iloc[idx]
            did = row["id_destino"]
            eta_horas = acumulado_km.get(did, km) / max(vel_media, 1e-6)
            eta_ts = row["ready_ts"] + pd.to_timedelta(eta_horas, unit="h")
            deadline = row["fecha_caducidad_pedido"]
            ok = pd.to_datetime(eta_ts) <= pd.to_datetime(deadline)
            cad_ok = cad_ok and bool(ok)
            pedidos_ids.append(row["id_pedido"])

        rutas.append({
            "vehiculo": vehiculo,
            "secuencia": sec,
            "carga_total": int(carga),
            "km": float(round(km, 2)),
            "tiempo_h": float(round(tiempo_h, 2)),
            "coste": float(round(coste, 2)),
            "caducidad_ok": bool(cad_ok),
            "pedidos": pedidos_ids,
        })
        vehiculo += 1

        if len(usados) >= len(pedidos_sorted):
            break

    # KPI globales
    km_tot = sum(r["km"] for r in rutas)
    t_tot = sum(r["tiempo_h"] for r in rutas)
    veh = len(rutas)
    cad_all = all(r["caducidad_ok"] for r in rutas)

    soluciones = [{
        "id": "heuristica_v1",
        "rutas": rutas,
        "kpi": {
            "km_totales": km_tot,
            "vehiculos": veh,
            "tiempo_total_h": t_tot,
            "caducidad_ok": cad_all
        }
    }]
    return soluciones
