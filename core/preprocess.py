# core/preprocess.py
# -*- coding: utf-8 -*-
# Preparación de datos de pedidos y cálculo de ready/deadline

import pandas as pd

def preparar_datos(pedidos: pd.DataFrame, productos: pd.DataFrame, destinos: pd.DataFrame):
    """
    Devuelve:
      - agg (por id_pedido) con:
          fecha_pedido (datetime)
          id_cliente
          id_destino
          qty_total
          ready_ts  = max(fecha_pedido + t_fabricacion) entre líneas del pedido
          fecha_caducidad_pedido = min(fecha_pedido + t_fabricacion + caducidad) entre líneas del pedido
      - productos (copia)
      - destinos  (copia)
    """
    pedidos = pedidos.copy()
    productos = productos.copy()
    destinos = destinos.copy()

    pedidos["fecha_pedido"] = pd.to_datetime(pedidos["fecha_pedido"])

    prod = productos.set_index("id_producto")[["tiempo_fabricacion_dias", "caducidad_dias"]]
    pedidos = pedidos.join(prod, on="id_producto", how="left")

    # Ready por línea
    pedidos["line_ready"] = pedidos["fecha_pedido"] + pd.to_timedelta(pedidos["tiempo_fabricacion_dias"], unit="D")

    # Deadline por línea
    pedidos["line_deadline"] = (
        pedidos["fecha_pedido"]
        + pd.to_timedelta(pedidos["tiempo_fabricacion_dias"], unit="D")
        + pd.to_timedelta(pedidos["caducidad_dias"], unit="D")
    )

    # Agregación por pedido
    agg = pedidos.groupby("id_pedido").agg(
        fecha_pedido=("fecha_pedido", "min"),
        id_cliente=("id_cliente", "first"),
        id_destino=("id_destino", "first"),
        qty_total=("cantidad", "sum"),
        ready_ts=("line_ready", "max"),
        fecha_caducidad_pedido=("line_deadline", "min"),
    ).reset_index()

    return agg, productos, destinos
