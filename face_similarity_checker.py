import cv2
import datetime
import dlib
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Dlib의 얼굴 감지기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # 압축 풀린 모델 파일 경로

previous_landmarks = None  # 이전 사진의 랜드마크 좌표를 저장할 변수

# 얼굴 랜드마크 추출 함수
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 이미지를 Grayscale로 변환
    faces = detector(gray)  # 얼굴 감지
    if len(faces) == 0:  # 얼굴이 인식 안 되면 "Can't find a face." 문구 출력
        print("Can't find a face.")
        return None
    
    for face in faces:
        landmarks = predictor(gray, face)
        points = []
        for n in range(0, 68):  # 68개의 랜드마크 포인트
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append((x, y))
        return points

# 랜드마크 점을 이미지 위에 그리기
def draw_landmarks(image, points, color=(0, 255, 0)):  # 초록색 랜드마크 표시
    for (x, y) in points:
        cv2.circle(image, (x, y), 2, color, -1)  # 각 랜드마크에 작은 점(원) 그리기

# 두 랜드마크 좌표 세트 간의 유사도를 계산하는 함수
def calculate_similarity(landmarks1, landmarks2):
    if len(landmarks1) != len(landmarks2):
        return 0  # 좌표 개수가 다르면 일치율 0

    distances = [np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in zip(landmarks1, landmarks2)]
    mean_distance = np.mean(distances)  # 평균 거리 계산

    # 최대 거리를 가정하여 일치율을 0~100%로 변환 (거리가 작을수록 일치율이 높음)
    max_distance = 100  # 가정한 최대 거리
    similarity = max(0, 100 - (mean_distance / max_distance * 100))  # 일치율 계산
    return similarity

# 팝업 창을 띄우는 함수 (일치도 메시지)
def show_similarity_popup(similarity):
    # tkinter 팝업 창 생성
    root = tk.Tk()
    root.withdraw()  # 팝업 창을 숨겨둠 (기본 창 숨김)
    messagebox.showinfo("Similarity", f"얼굴 일치율: {similarity:.2f}%")
    root.destroy()  # 팝업 창 종료 후, 메모리에서 해제

# 팝업 창을 띄우는 함수 (첫 번째 사진 메시지)
def show_message_popup(message):
    # tkinter 팝업 창 생성
    root = tk.Tk()
    root.withdraw()  # 팝업 창을 숨겨둠 (기본 창 숨김)
    messagebox.showinfo("Message", message)
    root.destroy()  # 팝업 창 종료 후, 메모리에서 해제q

#타원형 얼굴 가이드 틀 그리기 함수
def draw_ellipse_guide(image):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)  #타원의 중심을 화면의 가운데로 설정
    axes = (int(width * 0.2), int(height * 0.35))  # 원의 크기 설정 (x축, y축 반지름)
    cv2.ellipse(image, center, axes, 0, 0, 360, (255, 0, 0), 2)  #타원형 가이드 틀 그리기 (파란색)
    return image

capture = cv2.VideoCapture(0)  # 카메라 정보 받아오기

if not capture.isOpened():  # 카메라 연결 안 됨
    print("Can't open camera.")
    exit()

while True:
    ret, frame = capture.read()  # 프레임 읽기

    if not ret:  # 프레임을 읽을 수 없음 - 카메라 연결 확인
        print("Don't read frame")
        break

    # 타원형 가이드 틀 그리기
    frame_with_guide = draw_ellipse_guide(frame)

    cv2.imshow('Camera', frame_with_guide)  # 실시간 영상 화면에 표시

    key = cv2.waitKey(5) & 0xFF  # 키 입력 대기

    if key == ord('s'):  # 's'를 눌러 사진 촬영
        file_name = f"selfie_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"  # 파일 이름
        cv2.imwrite(file_name, frame)
        print(f"The photo is saved: {file_name}")

        # 랜드마크 추출
        landmarks = get_landmarks(frame)
        if landmarks:
            draw_landmarks(frame, landmarks)  # 랜드마크를 사진에 표시
            cv2.imshow('Landmarks', frame)
            cv2.waitKey(0)  # 키 입력 대기
            cv2.destroyWindow('Landmarks')

            if previous_landmarks:  # 이전 사진의 랜드마크가 있으면 비교
                similarity = calculate_similarity(previous_landmarks, landmarks)
                print(f"얼굴 일치율: {similarity:.2f}%")

                # 일치도를 팝업 창으로 띄우기
                show_similarity_popup(similarity)
            else:
                # 첫 번째 사진 메시지 팝업 띄우기
                show_message_popup("첫 번째 사진입니다. 일치율을 계산하려면 두 번째 사진을 촬영하세요.")

            previous_landmarks = landmarks  # 현재 랜드마크를 이전 랜드마크로 저장

    elif key == ord('q'):  # 'q'를 눌러 카메라 종료
        break

capture.release()  # 카메라 장치에서 받아온 메모리 해제
cv2.destroyAllWindows()  # 모든 창 제거
