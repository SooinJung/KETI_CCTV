import cv2

static_cam = 'rtsp://root:ketiabcs@192.168.0.120/axis-media/media.amp'
cap1 = cv2.VideoCapture(static_cam)

while True:
    ret1, frame1 = cap1.read()
    if not ret1:
        print("프레임을 가져오지 못했습니다.")
        break
    cv2.imshow("CCTV", frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cv2.destroyAllWindows()
