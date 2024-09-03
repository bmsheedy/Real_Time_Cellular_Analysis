from ultralytics import YOLO
from numba import jit


@jit
def m():
    model = YOLO("yolov8n.pt")
    model.train(data="data.yml", epochs=50)


m()
