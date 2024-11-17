from ultralytics import YOLO
import cv2 as cv


# Definição do modelo
model = YOLO(r"C:\Users\lusca\Universidade\CV\TPs\TP3_A82451\TP3_A82451\yolov8n.pt")

results = model(r"C:\Users\lusca\Universidade\CV\TPs\TP3_A82451\TP3_A82451\imgs\cuca_eu.jpeg", device="cpu")
# Iterar sobre os resultados
for result in results:
    # Gera a imagem anotada com caixas e rótulos
    annotated_image = result.plot()

    # Renderiza usando o OpenCV
    cv.imshow("análise YOLO", annotated_image)

    # Espera até que uma tecla seja pressionada e fecha a janela
    cv.waitKey(0)
    cv.destroyAllWindows()
