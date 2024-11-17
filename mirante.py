import cv2 as cv # type: ignore
from ultralytics import YOLO # type: ignore

# Modelo
MODELO = YOLO(r"C:\Users\lusca\Universidade\CV\TPs\TP3_A82451\TP3_A82451\yolov8n.pt")
#
#results = MODELO(r"C:\Users\lusca\Universidade\CV\TPs\TP3_A82451\TP3_A82451\imgs\cuca_eu.jpeg", device="cpu")

class Mirante:
    def __init__(self, origem):
        self.captura = cv.VideoCapture(origem)
        self.captura.set(cv.CAP_PROP_FRAME_WIDTH, 480)
        self.captura.set(cv.CAP_PROP_FRAME_HEIGHT, 320)
        self.captura.set(cv.CAP_PROP_FPS, 30)
        self.background_subtractor = cv.createBackgroundSubtractorMOG2(varThreshold=200)
        self.yolo = MODELO
    
    def colocar_imagem(self, frame, overlay_image, x_min, y_min, x_max, y_max):
        overlay_resized = cv.resize(overlay_image, (x_max - x_min, y_max - y_min)) # Ajustar o tamanho da imagem para caber na bounding box
        frame[y_min:y_max, x_min:x_max] = overlay_resized
            
    def analisar_frame(self, frame):
        # Vou chamar aqui as analises de movimento e face
        return self.detectar_movimento(frame)
    
    def detectar_objetos(self, frame):
        # Detecta objetos no frame
        results = self.yolo(frame)

        # Carregar as imagens que você deseja sobrepor
        cup_image = cv.imread(r"C:\Users\lusca\Universidade\CV\TPs\TP3_A82451\TP3_A82451\modelos\suarez.jpeg", cv.IMREAD_UNCHANGED)
        person_image = cv.imread(r"C:\Users\lusca\Universidade\CV\TPs\TP3_A82451\TP3_A82451\modelos\suarez.jpeg", cv.IMREAD_UNCHANGED)
        keyboard_image = cv.imread(r"C:\Users\lusca\Universidade\CV\TPs\TP3_A82451\TP3_A82451\modelos\suarez.jpeg", cv.IMREAD_UNCHANGED)  
        
        for result in results:
            boxes = result.boxes.xyxy
            classes = result.boxes.cls

            for i in range(len(boxes)):
                class_id = int(classes[i])
                class_name = result.names[class_id]

                # Obter as coordenadas da bounding box
                x_min, y_min, x_max, y_max = map(int, boxes[i])

                if class_name.lower() == "cup":
                    self.colocar_imagem(frame, cup_image, x_min, y_min, x_max, y_max)
                elif class_name.lower() == "person":
                    self.colocar_imagem(frame, person_image, x_min, y_min, x_max, y_max)
                elif class_name.lower() == "keyboard":
                    self.colocar_imagem(frame, keyboard_image, x_min, y_min, x_max, y_max)

        return frame

    
    def detectar_movimento(self, frame):
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur_frame = cv.GaussianBlur(gray_frame, (25, 25), 0)
        move_frame = self.background_subtractor.apply(blur_frame)
        thresh = cv.threshold(move_frame, 1, 255, cv.THRESH_BINARY)
        filled_frame = cv.dilate(thresh[1], None, iterations=25)

        frames = [frame, move_frame, filled_frame]
        return frames
        
    def mostrar(self, debug=False):
        frame_count = 0
        while self.captura.isOpened():
            ret, frame = self.captura.read()
            if not ret:
                print("Captura Interrompida")
                break
            
            frame_count += 1
            if frame_count % 2 != 0:  # pula 1 de 2
                continue
            
            frame_analisado = self.detectar_objetos(frame)
            # Validar frame antes de exibir
            if frame_analisado is None or frame_analisado.shape[0] == 0 or frame_analisado.shape[1] == 0:
                continue
            
            cv.imshow("Mirante", frame_analisado)
            
            if debug:
                frames = self.analisar_frame(frame) # Mudar isso depois, precisa voltar a ser chamado sempre essa linha pois sao os moves
                cv.imshow("Movimento", frames[1])
                cv.imshow("Binario", frames[2])
            
            key = cv.waitKey(1)
            keys = [27, ord("q"), ord("l")]
            if key in keys:
                break
            
        self.captura.release()
        cv.destroyAllWindows()
        
if __name__ == "__main__":
    mirante = Mirante(0)
    mirante.mostrar()