# -*- coding: utf-8 -*-
# Función de puntuación y ranking de soluciones
import pandas as pd

def evaluar_solutions(soluciones, alpha=1.0, beta=0.3, gamma=2.0):
    filas = []
    for s in soluciones:
        km = s["kpi"]["km_totales"]
        veh = s["kpi"]["vehiculos"]
        penal = 0.0 if s["kpi"].get("caducidad_ok", True) else 1.0
        score = alpha*km + beta*veh + gamma*penal
        filas.append({"id": s["id"], "km": km, "vehiculos": veh, "penal": penal, "score": score})
    df = pd.DataFrame(filas).sort_values("score").reset_index(drop=True)
    return df