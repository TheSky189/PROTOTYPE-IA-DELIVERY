# -*- coding: utf-8 -*-
# Heurística sencilla para planificar rutas con capacidad y caducidad
import pandas as pd

# Estructura de una solución (sugerida):
# {
#   "id": "sol_1",
#   "rutas": [
#       {"vehiculo": 1, "secuencia": ["Mataro", "D001", "D005", "Mataro"], "carga_total": 300,
#        "km": 420.0, "tiempo_h": 6.0, "caducidad_ok": True, "pedidos": ["P001","P002"]},
#       ...
#   ],
#   "kpi": {"km_totales": 900.0, "vehiculos": 3, "tiempo_total_h": 14.0, "caducidad_ok": True}
# }

def planificar_rutas(pedidos: pd.DataFrame, productos: pd.DataFrame, destinos: pd.DataFrame,
                     capacidad_camion: int, vel_media: float, coste_km: float, max_camiones: int):
    # TODO: implementar: agrupación simple por proximidad/ciudad y por capacidad; luego unir destinos
    # maqueta de salida mínima viable
    soluciones = [
        {
            "id": "solucion_demo",
            "rutas": [
                {"vehiculo": 1, "secuencia": ["Mataro", "D001", "Mataro"], "carga_total": 160,
                 "km": 66.0, "tiempo_h": 1.0, "caducidad_ok": True, "pedidos": ["P001"]},
                {"vehiculo": 2, "secuencia": ["Mataro", "D005", "Mataro"], "carga_total": 60,
                 "km": 600.0, "tiempo_h": 8.5, "caducidad_ok": True, "pedidos": ["P002"]},
            ],
            "kpi": {"km_totales": 666.0, "vehiculos": 2, "tiempo_total_h": 9.5, "caducidad_ok": True}
        }
    ]
    return soluciones