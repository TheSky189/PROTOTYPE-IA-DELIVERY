from core.preprocess import preparar_datos
import pandas as pd

def test_caducidad_ready_minmax():
    pedidos = pd.DataFrame([
        ["P001","2025-01-01","C1","D001","PR01",100],
        ["P001","2025-01-01","C1","D001","PR02",100],
    ], columns=["id_pedido","fecha_pedido","id_cliente","id_destino","id_producto","cantidad"])
    productos = pd.DataFrame([
        ["PR01","A",1,4],
        ["PR02","B",0,10]
    ], columns=["id_producto","nombre","tiempo_fabricacion_dias","caducidad_dias"])
    destinos = pd.DataFrame([["D001","X",41.38,2.17]], columns=["id_destino","nombre","lat","lon"])
    agg,_,_ = preparar_datos(pedidos, productos, destinos)
    row = agg.iloc[0]
    assert pd.to_datetime(row["ready_ts"]) >= pd.to_datetime(row["fecha_pedido"])
    assert pd.to_datetime(row["fecha_caducidad_pedido"]) >= pd.to_datetime(row["ready_ts"])
