import cv2
import numpy as np
import cx_Oracle as ora
import datetime

# 현재 시간을 가져와서 포맷팅
current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y-%m-%d %H:%M')

# Oracle 데이터베이스에 연결하기 위한 정보
dsn = ora.makedsn(host='localhost', port=1521, service_name='xe')
user = 'hr'
password = 'hr'

# 데이터베이스 연결
connection = ora.connect(user=user, password=password, dsn=dsn)

# 커서 생성
cursor = connection.cursor()

# 얼굴 탐지 모델 파일 경로
YOLO_CONFIG = 'C:\\project\\CCTV\\face.cfg'
YOLO_WEIGHTS = 'C:\\project\\CCTV\\face.weights'

# 성별 예측 모델 파일 경로
GENDER_MODEL = 'C:\\project\\CCTV\\weights\\deploy_gender.prototxt'
GENDER_PROTO = 'C:\\project\\CCTV\\weights\\gender_net.caffemodel'

# 입력 이미지 전처리를 위한 평균값
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# 성별 클래스 목록
GENDER_LIST = ['Male', 'Female']

# 클래스 목록
CLASSES = ["person"]

# 나이 예측 모델 파일 경로
AGE_MODEL = 'C:\\project\\CCTV\\weights\\deploy_age.prototxt'
AGE_PROTO = 'C:\\project\\CCTV\\weights\\age_net.caffemodel'

# 나이 구간 목록
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

# 프레임의 너비와 높이 초기화
frame_width = 1280
frame_height = 720

# 얼굴 탐지 모델 불러오기
face_net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)

# 나이 예측 모델 불러오기
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)

# 성별 예측 모델 불러오기
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

# 얼굴 탐지에 사용할 최소 확신도
CONFIDENCE_THRESHOLD = 0.5
# NMS 임계값
NMS_THRESHOLD = 0.4


def get_faces(frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    face_net.setInput(blob)
    layer_names = face_net.getUnconnectedOutLayersNames()
    outs = face_net.forward(layer_names)

    boxes = []
    confidences = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > CONFIDENCE_THRESHOLD:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                start_x = center_x - width // 2
                start_y = center_y - height // 2
                end_x = start_x + width
                end_y = start_y + height

                if width <= 0 or height <= 0:
                    print(f"Invalid face image size: ({width},{height})")
                    continue

                boxes.append([start_x, start_y, width, height])
                confidences.append(float(confidence))
    # NMS를 적용하여 겹치는 박스를 제거
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    faces = [boxes[i] for i in indices]

    # 얼굴이 인식이 제대로 이루어지는지 출력
    print(f"Detected {len(faces)} faces")
    
    return faces


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # 이미지 크기를 조정할 초기화
    dim = None
    (h, w) = image.shape[:2]

    # 너비와 높이가 모두 None인 경우 원본 이미지 반환
    if width is None and height is None:
        return image

    # 너비가 None인 경우
    if width is None:

        # 높이의 비율을 계산하고 차원 구성
        r = height / float(h)
        dim = (int(w * r), height)

    # 그렇지 않으면 높이가 None인 경우
    else:
        # 너비의 비율을 계산하고 차원 구성
        r = width / float(w)
        dim = (width, int(h * r))

    # 이미지 크기 조정
    return cv2.resize(image, dim, interpolation=inter)


def get_gender_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    gender_net.setInput(blob)
    return gender_net.forward()


def get_age_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False
    )
    age_net.setInput(blob)
    return age_net.forward()


def box_distance(box1, box2):
    center1 = (box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2
    center2 = (box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)


def main():
    accumulated_faces = {}
    face_id = 0
    left_zone_count = 0
    right_zone_count = 0
    total_count = 0

    def predict_age_and_gender_webcam(show_result: bool = False, left_zone_count=0, right_zone_count=0,
                                      accumulated_faces=None, face_id=0):
        if accumulated_faces is None:
            accumulated_faces = {}

        left_zone_count
        right_zone_count
        accumulated_faces
        face_id
        # 웹캠을 사용하도록 설정합니다. 비디오 파일 대신 웹캠을 엽니다.
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        if not cap.isOpened():
            print("Error: Unable to open webcam")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            print(frame.shape)
            print(frame[0, 0])

            if not ret:
                break

            # 세로선 그리기 (화면 중앙)
            cv2.line(frame, (frame_width // 2, 0), (frame_width // 2, frame_height), (0, 255, 0), 1)

            if frame.shape[1] > frame_width:
                frame = image_resize(frame, width=frame_width)

            faces = get_faces(frame)

            if not faces:
                print("NO faces detected.")
                continue

            for i, (start_x, start_y, width, height) in enumerate(faces):
                end_x = start_x + width
                end_y = start_y + height

                print(start_x, start_y, end_x, end_y)
                if width <= 0 or height <= 0 or end_x > frame.shape[1] or end_y > frame.shape[0]:
                    print(f"Invalid coordinates for face {i}: ({start_x}, {start_y}), ({end_x}, {end_y}), size: ({width}, {height})")
                    continue
                face_img = frame[start_y:end_y, start_x:end_x]
                if face_img.shape[0] <= 0 or face_img.shape[1] <= 0:
                    print(f"Invalid face image size for face {i}: {face_img.shape}")
                    continue
                # 이미 처리한 얼굴인지 확인

                face_box = [start_x, start_y, end_x, end_y]

                face_processed = False
                for fid, face_data in accumulated_faces.items():
                    if box_distance(face_box, face_data["box"]) < 50:
                        face_processed = True
                        break
                if not face_processed:
                    face_id += 1
                    accumulated_faces[face_id] = {
                        "box":face_box,
                        "counted":False
                    }
                # 얼굴이 이미 카운트되었는지 확인
                if not accumulated_faces[face_id]["counted"]:
                    if (end_x + start_x) / 2 < frame_width / 2:
                        left_zone_count += 1
                    else:
                        right_zone_count += 1
                    total_count = left_zone_count + right_zone_count
                    accumulated_faces[face_id]["counted"] = True

                

                if face_img.shape[0] <- 0 or face_img.shape[1] <= 0:
                    print(f"Invalid face imgae size : {face_img.shape}")
                    continue
                print(face_img.shape)

                age_preds = get_age_predictions(face_img)
                gender_preds = get_gender_predictions(face_img)

                # 성별과 나이 가져오기
                i = gender_preds[0].argmax()
                gender = GENDER_LIST[i]
                gender_confidence_score = gender_preds[0][i]
                
                age_prediction = age_preds[0]
                age_index = age_prediction.argmax()
                age_confidence_score = age_prediction[age_index]
                age = AGE_INTERVALS[age_index]

                # 프레임에 정보 표시
                label = f"{gender}-{gender_confidence_score * 100:.1f}%, {age}-{age_confidence_score * 100:.1f}%"
                yPos = start_y - 15
                while yPos < 15:
                    yPos += 15
                box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
                font_scale = 0.54
                cv2.putText(frame, label, (start_x, yPos),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 2)

                # 각 얼굴에 고유 ID 표시
                cv2.putText(frame, f"ID: {face_id}", (start_x, end_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.54, box_color, 2)

            # 프레임에 영역 카운트 텍스트 추가
            zone_count_text = f"left_count : {left_zone_count} right_count : {right_zone_count}"
            cv2.putText(frame, zone_count_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if show_result:
                cv2.imshow("Result Video", frame)
                # 'ESC' 키를 눌러 창을 닫습니다.
                if cv2.waitKey(20) & 0xFF == 27:
                    break
        print(f"left_count : {left_zone_count} right_count : {right_zone_count}")

        # 데이터베이스에 데이터 추가
        query = f"INSERT INTO face_num (total, count_date) VALUES ({left_zone_count + right_zone_count}, TO_TIMESTAMP('{formatted_time}', 'YYYY-MM-DD HH24:MI'))"
        print(query)
        cursor.execute(query)
        # 커밋 작업
        connection.commit()
        # 커서와 연결 해제 
        cursor.close()
        # 연결 해제
        connection.close()
        cap.release()
        cv2.destroyAllWindows()
        return left_zone_count, right_zone_count

    # 웹캠을 사용하여 얼굴 인식 결과 화면을 표시
    left_zone_count, right_zone_count = predict_age_and_gender_webcam(show_result=True, left_zone_count=left_zone_count,
                                                                      right_zone_count=right_zone_count,
                                                                      accumulated_faces=accumulated_faces,
                                                                      face_id=face_id)

if __name__ == "__main__":
    main()

