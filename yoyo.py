import cv2 as cv
from ultralytics import YOLO

# Modelo
MODELO = YOLO(r"C:\Users\lusca\Universidade\CV\TPs\TP3_A82451\TP3_A82451\yolov8n.pt")

#results = MODELO(source="https://www.youtube.com/jFQnvkO3UtE&t", device="cpu", stream=True, show=True)
#results = MODELO(source=0, show=True)
captura = cv.VideoCapture(r".\imgs\luanfazendogol.mp4")
captura.set(cv.CAP_PROP_FRAME_WIDTH, 640)
captura.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
captura.set(cv.CAP_PROP_FPS, 60)
background_subtractor = cv.createBackgroundSubtractorMOG2(varThreshold=200)
while captura.isOpened():
    ret, frame = captura.read()
    if not ret:
        print("Captura Interrompida")
        break
    
    results = MODELO(frame)
    frame_analisado = results[0].plot()
    cv.imshow("YoYo Testes", frame_analisado)
    #detectar_objetos(frame)
    
    key = cv.waitKey(1)
    keys = [27, ord("q"), ord("l")]
    if key in keys:
        break
    
captura.release()
cv.destroyAllWindows()