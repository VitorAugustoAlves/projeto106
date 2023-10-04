import cv2

body_classifier_path = './haarcascade_fullbody.xml'

body_classifier = cv2.CascadeClassifier(body_classifier_path)

# Inicie a captura de vídeo para o arquivo de vídeo
cap = cv2.VideoCapture('walking.avi')

# Faça o loop assim que o vídeo for carregado com sucesso
while True:
    
    # Leia o primeiro quadro
    ret, frame = cap.read()

    # Converta cada quadro em escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Passe o quadro para nosso classificador de corpos
    bodies = body_classifier.detectMultiScale(gray, 1.1, 3)

    # Inicialize a lista para armazenar as caixas delimitadoras
    bounding_boxes = []

    # Desenha retângulos ao redor dos corpos identificados e armazena as caixas delimitadoras
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        bounding_boxes.append((x, y, x+w, y+h))

    # Exibe o frame resultante
    cv2.imshow('Detecção de Corpos', frame)

    # Exibe as caixas delimitadoras na lista
    print("Caixas Delimitadoras:", bounding_boxes)

    # Se a tecla 'q' for pressionada, saia do loop
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
