# core/vrp_savings.py
# -*- coding: utf-8 -*-
# Clarke–Wright Savings con capacidad (versión simplificada para prototipo)

import pandas as pd
import networkx as nx
from .graph import construir_grafo, ORIGEN_ID

def _ruta_km(G: nx.Graph, seq):
    km = 0.0
    for i in range(len(seq)-1):
        u, v = seq[i], seq[i+1]
        km += G[u][v]["weight"]
    return km

def planificar_rutas_savings(pedidos_df: pd.DataFrame, destinos_df: pd.DataFrame,
                             capacidad_camion: int, vel_media: float, coste_km: float, max_camiones: int):
    """
    Versión 'capacity-aware' simple:
      1) Inicializa rutas como [O, d, O] para cada destino.
      2) Calcula 'ahorros' S(i,j) = c(O,i)+c(O,j)-c(i,j).
      3) Intenta fusionar rutas en los extremos si no excede capacidad.
    """
    G = construir_grafo(destinos_df)

    # Agregar por destino (sumar qty_total de pedidos con el mismo id_destino)
    by_dest = (pedidos_df.groupby("id_destino")["qty_total"]
               .sum().reset_index().sort_values("id_destino"))
    demanda = {row["id_destino"]: int(row["qty_total"]) for _, row in by_dest.iterrows()}
    destinos = list(demanda.keys())

    # 1) rutas iniciales (estrella)
    rutas = []
    for d in destinos:
        rutas.append({
            "secuencia": [ORIGEN_ID, d, ORIGEN_ID],
            "carga_total": demanda[d],
        })

    # 2) savings
    savings = []
    for i in range(len(destinos)):
        for j in range(i+1, len(destinos)):
            di, dj = destinos[i], destinos[j]
            if not (G.has_edge(ORIGEN_ID, di) and G.has_edge(ORIGEN_ID, dj) and G.has_edge(di, dj)):
                continue
            s = G[ORIGEN_ID][di]["weight"] + G[ORIGEN_ID][dj]["weight"] - G[di][dj]["weight"]
            savings.append((di, dj, s))
    # Ordenar por ahorro descendente
    savings.sort(key=lambda x: x[2], reverse=True)

    # Helper: localizar ruta y si d es extremo
    def _find_route_with_endpoints(d):
        for idx, r in enumerate(rutas):
            seq = r["secuencia"]
            if len(seq) < 3:  # [O, O] improbable
                continue
            # extremos son los nodos justo antes del origen en cada lado
            # seq = [O, ..., X, O] -> extremos posibles X y el primero después de O
            left_end = seq[1]
            right_end = seq[-2]
            if d == left_end or d == right_end:
                return idx, left_end, right_end
        return None, None, None

    # 3) intentos de fusión
    for (di, dj, _) in savings:
        i_idx, i_left, i_right = _find_route_with_endpoints(di)
        j_idx, j_left, j_right = _find_route_with_endpoints(dj)
        if i_idx is None or j_idx is None or i_idx == j_idx:
            continue

        # Intentar unir por extremos compatibles
        # Caso A: ... di ]O + [O dj ...
        def _try_merge(i_idx, j_idx, a_end, b_end):
            ri = rutas[i_idx]["secuencia"]
            rj = rutas[j_idx]["secuencia"]
            carga = rutas[i_idx]["carga_total"] + rutas[j_idx]["carga_total"]
            if carga > capacidad_camion:
                return False
            # Queremos ri: [O ... a_end O], rj: [O b_end ... O]
            # y fusionar en [O ... a_end , ... b_end ... O] sin O intermedio.
            # Para eso, invertimos si hace falta para alinear extremos.
            def ends(seq):
                return seq[1], seq[-2]
            ai, bi = ends(ri)
            aj, bj = ends(rj)

            # orientar ri para que su extremo de unión sea 'a_end' en la cola
            if ri[1] == a_end:
                ri_core = ri[1:-1]  # [a_end, ...]
            else:
                ri_core = list(reversed(ri[1:-1]))  # invertimos

            # orientar rj para que su extremo de unión sea 'b_end' en la cabeza
            if rj[-2] == b_end:
                rj_core = rj[1:-1]  # [..., b_end]
            else:
                rj_core = list(reversed(rj[1:-1]))

            nueva = [ORIGEN_ID] + ri_core + rj_core + [ORIGEN_ID]
            rutas[i_idx] = {"secuencia": nueva, "carga_total": carga}
            rutas.pop(j_idx)
            return True

        # Cuatro combinaciones de extremos: (i_right con j_left) y variantes
        merged = (
            _try_merge(i_idx, j_idx, i_right, j_left) or
            _try_merge(i_idx, j_idx, i_left, j_right) or
            _try_merge(j_idx, i_idx, j_right, i_left) or
            _try_merge(j_idx, i_idx, j_left, i_right)
        )
        if merged and len(rutas) <= max_camiones:
            # opcionalmente, seguir; si se quisiera estrictamente <= max_camiones, podríamos cortar aquí
            pass

    # KPI por ruta y global
    result_rutas = []
    for k, r in enumerate(rutas, start=1):
        km = _ruta_km(G, r["secuencia"])
        t_h = km / max(vel_media, 1e-6)
        coste = km * coste_km
        result_rutas.append({
            "vehiculo": k,
            "secuencia": r["secuencia"],
            "carga_total": int(r["carga_total"]),
            "km": float(round(km, 2)),
            "tiempo_h": float(round(t_h, 2)),
            "coste": float(round(coste, 2)),
            # caducidad_ok: no calculamos aquí por pedido; lo dejamos al comparador si hace falta
            "caducidad_ok": True,
            "pedidos": []  # en savings trabajamos a nivel destino; rellenar opcional
        })

    km_tot = sum(x["km"] for x in result_rutas)
    t_tot = sum(x["tiempo_h"] for x in result_rutas)
    soluciones = [{
        "id": "savings_cw",
        "rutas": result_rutas,
        "kpi": {
            "km_totales": km_tot,
            "vehiculos": len(result_rutas),
            "tiempo_total_h": t_tot,
            "caducidad_ok": all(r["caducidad_ok"] for r in result_rutas)
        }
    }]
    return soluciones
