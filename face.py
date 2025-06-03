import cv2

face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
video_path = 'my_video.mp4'
cap = cv2.VideoCapture(video_path)

# Перевірка, чи відеофайл успішно відкрито
if not cap.isOpened():
    print(f"Помилка: Не вдалося відкрити відеофайл за шляхом: {video_path}")
else:
    while True:
        # Читаємо кадр за кадром з відео
        success, img = cap.read()

        # Якщо кадри закінчилися (кінець відео) або не вдалося прочитати кадр
        if not success:
            print("Кінець відео або помилка читання кадру.")
            break

        # Перетворюємо зображення у відтінки сірого (покращує ефективність розпізнавання)
        # [cite: 15, 22]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Виявляємо обличчя на кадрі
        # Параметри detectMultiScale:
        #   img_gray: зображення у відтінках сірого
        #   1.1: scaleFactor - наскільки зменшується розмір зображення на кожному масштабі зображення
        #   19: minNeighbors - скільки сусідів повинен мати кожен кандидат-прямокутник, щоб його зберегти
        # Ці параметри можуть потребувати налаштування для кращої точності
        # [cite: 18, 21, 22]
        faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 19)

        # Малюємо прямокутники навколо виявлених облич
        # [cite: 19, 21, 22]
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) # Зелений прямокутник товщиною 2px

        # Відображаємо результат
        cv2.imshow('Video Face Detection', img)

        # Умова для виходу з циклу (натискання клавіші 'q')
        # [cite: 22]
        if cv2.waitKey(1) & 0xFF == ord('q'): # Очікуємо 1 мс, якщо натиснуто 'q' - вихід
            break

# Звільняємо об'єкт захоплення відео та закриваємо всі вікна OpenCV
cap.release()
cv2.destroyAllWindows()

print("Обробка відео завершена.")