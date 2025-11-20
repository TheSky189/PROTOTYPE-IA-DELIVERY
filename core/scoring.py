# core/scoring.py
# -*- coding: utf-8 -*-
# Comparador y ranking de soluciones de varios modelos

import pandas as pd

def evaluar_solutions(solutions, alpha=1.0, beta=0.3, gamma=2.0):
    """
    Crea un ranking con una puntuación simple:
      score = alpha * km + beta * vehiculos + gamma * penalización_caducidad
    Menor score es mejor.
    """
    filas = []
    for s in solutions:
        k = s["kpi"]
        pen_cad = 0.0 if k.get("caducidad_ok", True) else 1000.0
        score = alpha * k["km_totales"] + beta * k["vehiculos"] + gamma * pen_cad
        filas.append({
            "id": s["id"],
            "km_totales": k["km_totales"],
            "vehiculos": k["vehiculos"],
            "tiempo_total_h": k["tiempo_total_h"],
            "caducidad_ok": k.get("caducidad_ok", True),
            "score": round(score, 3)
        })
    df = pd.DataFrame(filas).sort_values("score", ascending=True).reset_index(drop=True)
    return df
