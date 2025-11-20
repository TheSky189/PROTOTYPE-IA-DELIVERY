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
def calcular_rutas():
    """
    Función de ejemplo que devuelve rutas predefinidas para probar
    la integración con Streamlit + mapa Folium.

    Retorna una lista de rutas, donde cada ruta contiene:
    - número de camión
    - lista de paradas con nombre, latitud y longitud
    """

    rutas = [
        {
            "camion": 1,
            "paradas": [
                {"nombre": "Mataró", "lat": 41.5421, "lon": 2.4445},
                {"nombre": "Barcelona", "lat": 41.3874, "lon": 2.1686},
                {"nombre": "Lleida", "lat": 41.6176, "lon": 0.6200},
                {"nombre": "Mataró (Regreso)", "lat": 41.5421, "lon": 2.4445}  # opcional
            ]
        },
        {
            "camion": 2,
            "paradas": [
                {"nombre": "Mataró", "lat": 41.5421, "lon": 2.4445},
                {"nombre": "Tarragona", "lat": 41.1189, "lon": 1.2445},
                {"nombre": "Mataró (Regreso)", "lat": 41.5421, "lon": 2.4445}  # opcional
            ]
        }
    ]

    return rutas

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