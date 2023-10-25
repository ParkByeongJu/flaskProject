import cv2
import numpy as np
import cx_Oracle as db
import datetime

# 현재 시간을 가져와서 포맷팅
current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y-%m-%d %H:%M')

# 오라클 DB 연결
dsn = db.makedsn(host='localhost', port=1521, service_name='xe')
user = 'hr'
password = 'hr'
connection = db.connect(user=user, password=password, dsn=dsn)

# 커서 생성
cursor = connection.cursor()

# YOLO 얼굴 탐지 모델 파일 경로
YOLO_CFG = "C:\\project\\CCTV\\face.cfg"
YOLO_WEIGHTS = "C:\\project\\CCTV\\face.weights"

# 나이 예측 모델 파일 경로
AGE_MODEL = "C:\\project\\CCTV\\weights\\age_net.caffemodel"
AGE_PROTO = "C:\\project\\CCTV\\weights\\deploy_age.prototxt"

# 성별 예측 모델 파일 경로
GENDER_MODEL = "C:\\project\\CCTV\\weights\\gender_net.caffemodel"
GENDER_PROTO = "C:\\project\\CCTV\\weights\\deploy_gender.prototxt"

# YOLO 모델 불러오기
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)

# 성별 예측 모델 불러오기
gender_model = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

# 나이 예측 모델 불러오기
age_model = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)

# 출력 레이어 이름 가져오기
output_layers_names = net.getUnconnectedOutLayersNames()

# 나이 구간 목록
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# 성별 목록
gender_list = ["Male", "Female"]

# 입력 이미지 전처리를 위한 평균값
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# 얼굴을 검출하는 함수, 프레임과 신뢰도를 입력받음
def get_faces(frame, confidence_threshold=0.4):
    # 이미지를 딥러닝 모델이 이해할 수 있는 형태로 변환
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104, 177.0, 123.0))
    
    # blob을 바탕으로 얼굴인식 수행
    face_net.setInput(blob)
    
    # face_net모델이 얼굴인식을 수행 후 결과 반환하고, 
    # 이를 np.squeeze를 사용해서 1차원 형태로 변환
    output = np.squeeze(net.forward())
    
    # 얼굴 목록 초기화
    faces = []

    
    for i in range(output.shape[0]):
        # 각 얼굴에 대한 신뢰도 점수 획득
        confidence = output[i, 2]
        # 신뢰도 점수가 임계값보다 높은 경우에만 얼굴인식
        if confidence > confidence_threshold:
            # 경계 상자의 좌표를 실제 픽셀 좌표로 변환
            box = output[i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            
            # 좌표를 정수로 변환
            start_x, start_y, end_x, end_y = box.astype(int)
            
            # 경계 상자를 약간 확장
            start_x, start_y, end_x, end_y = start_x - 10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y

            # 감지된 얼굴을 리스트에 추가
            faces.append((start_x, start_y, end_x, end_y))
    return faces
