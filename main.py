import cv2

face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
url = "http://192.168.43.86:4747/video?640x480"
capti = cv2.VideoCapture(url)

while True:
    succ, img = capti.read()

    # img=cv2.imread("1.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fazi = face_cascade_db.detectMultiScale(img_gray, 1.1, 5)
    for (x, y, w, h) in fazi:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img_gray_fazio = img_gray[y:y + h, x:x + w]
        ochi = eye_cascade_db.detectMultiScale(img_gray_fazio, 1.1, 5)
        for (ex,ey,ew,eh) in ochi:
            cv2.rectangle(img, (x+ex, y+ey), (x+ex + ew, y+ey + eh), (255, 0, 0), 2)

    cv2.imshow("rezki", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

capti.release()
cv2.destroyAllWindows()
