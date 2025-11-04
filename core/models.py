# -*- coding: utf-8 -*-
# Modelos de datos simples para tipificar el flujo
from dataclasses import dataclass
from datetime import date

@dataclass
class Producto:
    id: str
    nombre: str
    precio: float
    t_fabricacion: int   # días
    caducidad: int       # días

@dataclass
class PedidoLinea:
    id_pedido: str
    fecha_pedido: date
    id_cliente: str
    id_destino: str
    id_producto: str
    cantidad: int

@dataclass
class Destino:
    id: str
    nombre: str
    lat: float | None
    lon: float | None
    dist_km_desde_mataro: float | None