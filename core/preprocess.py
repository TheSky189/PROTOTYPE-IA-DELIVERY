# -*- coding: utf-8 -*-
# Preparación de datos: cálculo de fecha de caducidad del pedido y validaciones
import pandas as pd
from datetime import timedelta

def preparar_datos(pedidos: pd.DataFrame, productos: pd.DataFrame, destinos: pd.DataFrame):
    """Devuelve dataframes limpios y un campo 'fecha_caducidad_pedido' por id_pedido
    tomando el mínimo (fecha_pedido + t_fabricacion + caducidad) entre sus líneas.
    """
    # Normalizar columnas esperadas
    pedidos = pedidos.copy()
    pedidos["fecha_pedido"] = pd.to_datetime(pedidos["fecha_pedido"]).dt.date

    prod = productos.set_index("id_producto")[
        ["tiempo_fabricacion_dias", "caducidad_dias"]
    ]

    # Unir tiempos de fabricación y caducidad por línea
    pedidos = pedidos.join(
        prod, on="id_producto", how="left"
    )

    # Calcular fecha de caducidad por línea
    pedidos["caducidad_linea"] = pd.to_datetime(pedidos["fecha_pedido"]) \
        + pd.to_timedelta(pedidos["tiempo_fabricacion_dias"], unit="D") \
        + pd.to_timedelta(pedidos["caducidad_dias"], unit="D")

    # Reducir a nivel pedido: mínimo entre sus líneas
    min_cad = pedidos.groupby("id_pedido")["caducidad_linea"].min().rename("fecha_caducidad_pedido")
    pedidos = pedidos.merge(min_cad, on="id_pedido", how="left")

    # Destinos: asegurar dist o lat/lon
    destinos = destinos.copy()
    return pedidos, productos, destinos