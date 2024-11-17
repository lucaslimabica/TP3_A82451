import cv2 as cv


# Propriedades do vídeo
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FPS, 24)

background_subtractor = cv.createBackgroundSubtractorMOG2(varThreshold=200)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur_frame = cv.GaussianBlur(gray_frame, (25, 25), 0)
    move_frame = background_subtractor.apply(blur_frame)
    thresh = cv.threshold(move_frame, 1, 255, cv.THRESH_BINARY)
    filled_frame = cv.dilate(thresh[1], None, iterations=25)
    
    cv.imshow(f"Webcam Movimentos", move_frame)
    cv.imshow(f"Webcam Movimentos Preenchidos", filled_frame)

    cv.imshow("Minha Webcam", frame)
    key = cv.waitKey(1)
    closer_keys = [27, ord("q"), ord("l")]
    if key in closer_keys:
        break

cap.release()
cv.destroyAllWindows()
