import cv2
import mediapipe as mp
import time
import numpy as np
import pygame
import threading
import math
import random


def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ = 180
    return angle_

# 根據傳入的 21 個節點座標，得到該手指的角度
def hand_angle(hand_):
    angle_list = []
    # thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    # index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    # middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    # ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    # pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

# 根據手指角度的串列內容，返回對應的手勢名稱
def hand_pos(finger_angle):
    f1 = finger_angle[0]   # 大拇指角度
    f2 = finger_angle[1]   # 食指角度
    f3 = finger_angle[2]   # 中指角度
    f4 = finger_angle[3]   # 無名指角度
    f5 = finger_angle[4]   # 小拇指角度

    # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮
    if f1<50 and f2>=50 and f3>=50 and f4>=50 and f5>=50:
        return 'good'
    elif f1>=50 and f2>=50 and f3>=50 and f4>=50 and f5>=50:
        return '0'
    elif f1>=50 and f2<50 and f3>=50 and f4>=50 and f5>=50:
        return '1'
    elif f1>=50 and f2<50 and f3<50 and f4>=50 and f5>=50:
        return '2'
    elif f1>=50 and f2<50 and f3<50 and f4<50 and f5>50:
        return '3'
    '''
    elif f1>=50 and f2<50 and f3<50 and f4<50 and f5<50:
        return '4'
    elif f1<50 and f2<50 and f3<50 and f4<50 and f5<50:
        return '5'
    elif f1<50 and f2>=50 and f3>=50 and f4>=50 and f5<50:
        return '6'
    elif f1<50 and f2<50 and f3>=50 and f4>=50 and f5>=50:
        return '7'
    elif f1<50 and f2<50 and f3<50 and f4>=50 and f5>=50:
        return '8'
    elif f1<50 and f2<50 and f3<50 and f4<50 and f5>=50:
        return '9'
    else:
        return ''    
    '''



def is_close(p1, p2, threshold=30):
    return np.linalg.norm(p1 - p2) < threshold

# 定義函數來檢查手指座標是否低於某個閾值
def is_below_threshold(coord, threshold):
    return coord > threshold

# 定義函數來播放音樂
def play_music(note):
    sound = pygame.mixer.Sound(note)
    sound.play()

# 定義按鈕的回調函數
def on_button_click(event, x, y, flags, param):
    global running, window_width, window_height
    if event == cv2.EVENT_LBUTTONDOWN:  # 如果左鍵按下
        print("Button clicked!")  # 在控制台輸出消息
        running = True  # 設置運行標誌為True
    elif event == cv2.EVENT_RBUTTONDOWN:  # 如果右鍵按下
        window_width = 800  # 修改視窗寬度
        window_height = 600  # 修改視窗高度

def rand_song(img):
    n_rand_song=random.randint(1,3)
    if n_rand_song==1:
        cv2.putText(img, f"song:{n_rand_song}", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        return music_song1
    elif n_rand_song==2:
        cv2.putText(img, f"song:{n_rand_song}", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        return music_song2
    elif n_rand_song==3:
        cv2.putText(img, f"song:{n_rand_song}", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        return music_song3
 

def whitch_note(music_song,where):
    if music_song[where]==1:
        return "Do4.wav"
    if music_song[where]==2:
        return "Re4.wav"
    if music_song[where]==3:
        return "Mi4.wav"
    if music_song[where]==4:
        return "Fa4.wav"
    if music_song[where]==5:
        return "So4.wav"
    if music_song[where]==6:
        return "La4.wav"
    if music_song[where]==7:
        return "Ti4.wav"
    if music_song[where]==-1:
        return "Do3.wav"
    if music_song[where]==-2:
        return "Re3.wav"
    if music_song[where]==-3:
        return "Mi3.wav"
    if music_song[where]==-4:
        return "Fa3.wav"
    if music_song[where]==-5:
        return "So3.wav"
    if music_song[where]==-6:
        return "La3.wav"
    if music_song[where]==-7:
        return "Ti3.wav"

def call_part(img,part):
    if part==1:
        cv2.putText(img, f"song:1~3", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    if part==2:
        cv2.putText(img, f"practice", (210, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    if part==3:
        cv2.putText(img, f"free practice", (425, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

# 定義影像處理函數
def process_image():
    global cap, hands, handLmsStyle, handConStyle, finger_states,music_song,music_song1,music_song2,music_song3,music_test

    num_points=0#分數
    open_close=0#判斷手指碰觸節點
    where_music_song=0
    part=0
    part_2=0
    music_song=[-1,-2,-3,-4,-5,-6,-7,1,2,3,4,5,6,7]
    music_song1=[1,1,5,5,6,6,5,4,4,3,3,2,2,1,5,5,4,4,3,3,2,5,5,4,4,3,3,2,1,1,5,5,6,6,5,4,4,3,3,2,2,1]
    music_test=[-1,-2,-3,-4,-5,-6,-7,1,2,3,4,5,6,7]#測試用
    music_song2=[5,5,3,1,5,5,3,1,2,3,4,4,3,4,5,5,5,3,5,3,2,3,1,4,2,2,2,3,1,1,1,2,3,4,2,1,-7,1]#火車快飛
    music_song3=[1,2,-7,1,2,-7,1,2,-7,1,-7,1,2,
    -6,-7,-5,-6,-7,-5,-6,-7,-5,-6,-7,
    1,2,-7,1,2,-7,1,2,-7,1,-7,1,2,
    -6,-7,-5,-6,-7,-5,-6,-7,
    -5,-6,-5,-6,3,2,-7,2,
    -6,-6,-6,-6,
    -6,-6,-6,-6,
    -6,-6,-6,-6,
    -5,-5,-5,-5,-5,-5]
    while True:
        ret, img = cap.read()
        if ret:
     
            # 水平翻轉影像
            img = cv2.flip(img, 1)

            # 將影像轉換為 RGB 格式
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 使用 MediaPipe 進行手部偵測
            result = hands.process(imgRGB)
            # 檢查是否有偵測到手部
            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    if part==1:#選歌曲
                        results = hands.process(img)                # 偵測手勢
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                finger_points = []                   # 記錄手指節點座標的串列
                                for i in hand_landmarks.landmark:
                                    # 將 21 個節點換算成座標，記錄到 finger_points
                                    x = i.x*window_width
                                    y = i.y*window_height
                                    finger_points.append((x,y))
                                if finger_points:
                                    finger_angle = hand_angle(finger_points) # 計算手指角度，回傳長度為 5 的串列
                                    #print(finger_angle)                     # 印出角度 ( 有需要就開啟註解 )
                                    text = hand_pos(finger_angle)
                                    if text=='0':
                                        music_song=rand_song(img) 
                                    if text=='1':
                                        music_song=music_song1
                                        cv2.putText(img, f"song:1", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                                    if text=='2':
                                        music_song=music_song2
                                        cv2.putText(img, f"song:2", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                                    if text=='3':
                                        music_song=music_song3
                                        cv2.putText(img, f"song:3", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    if part==2:#按照順序
                        # 取得手部的關節點資訊
                        handLms_np = np.array([[lm.x, lm.y, lm.z] for lm in handLms.landmark])
                        one_tip = handLms_np[4][1] * img.shape[0]
                        two_tip = handLms_np[8][1] * img.shape[0]
                        two_tip2 = handLms_np[8][1] * img.shape[1]
                        mid_tip = handLms_np[12][1] * img.shape[0]
                        mid_tip2 = handLms_np[12][1] * img.shape[1]
                        four_tip = handLms_np[16][1] * img.shape[0]
                        four_tip2 = handLms_np[16][1] * img.shape[1]
                        five_tip = handLms_np[20][1] * img.shape[0]
                        five_tip2 = handLms_np[20][1] * img.shape[1]

                        if is_close(one_tip, two_tip):
                            open_close=1
                            
                        elif is_close(one_tip, mid_tip):
                            open_close=2

                        elif is_close(one_tip, four_tip):
                            open_close=3

                        elif is_close(one_tip, five_tip):
                            open_close=4

                        
                        #回到音樂的起點
                        if where_music_song==len(music_song):
                                where_music_song=0

                        if not is_close(one_tip, two_tip) and open_close==1:
                            if part_2==0 or part_2==4:
                                num_points+=1
                                note=whitch_note(music_song,where_music_song)
                                threading.Thread(target=play_music, args=(note,)).start()
                                where_music_song+=1
                                open_close=0
                                part_2=1
                                #cv2.circle(img,(int(two_tip),int(two_tip2)),10, (0,0,255,),cv2.FILLED)
                            else:
                                note="wrong.wav"
                                threading.Thread(target=play_music, args=(note,)).start()
                                open_close=0
                        
                        elif not is_close(one_tip, mid_tip)and open_close==2:
                            if part_2==1:
                                num_points+=1
                                note=whitch_note(music_song,where_music_song)
                                threading.Thread(target=play_music, args=(note,)).start()
                                where_music_song+=1
                                open_close=0
                                part_2+=1
                                #cv2.circle(img,(int(mid_tip),int(mid_tip2)),10, (0,0,255,),cv2.FILLED)
                            else:
                                note="wrong.wav"
                                threading.Thread(target=play_music, args=(note,)).start()
                                open_close=0
                        elif not is_close(one_tip, four_tip)and open_close==3:
                            if part_2==2:
                                num_points+=1
                                note=whitch_note(music_song,where_music_song)
                                threading.Thread(target=play_music, args=(note,)).start()
                                where_music_song+=1
                                open_close=0
                                part_2+=1
                                #cv2.circle(img,(int(four_tip),int(four_tip2)),10, (0,0,255,),cv2.FILLED)
                            else:
                                note="wrong.wav"
                                threading.Thread(target=play_music, args=(note,)).start()
                                open_close=0

                        elif not is_close(one_tip, five_tip) and open_close==4:
                            if part_2==3:
                                num_points+=1
                                note=whitch_note(music_song,where_music_song)
                                threading.Thread(target=play_music, args=(note,)).start()
                                where_music_song+=1
                                open_close=0
                                part_2+=1
                                #cv2.circle(img,(int(five_tip),int(five_tip2)),10, (0,0,255,),cv2.FILLED)
                            else:
                                note="wrong.wav"

                                threading.Thread(target=play_music, args=(note,)).start()
                                open_close=0

                    if part==3:#自由遊玩

                        # 取得手部的關節點資訊
                        handLms_np = np.array([[lm.x, lm.y, lm.z] for lm in handLms.landmark])
                        one_tip = handLms_np[4][1] * img.shape[0]
                        two_tip = handLms_np[8][1] * img.shape[0]
                        mid_tip = handLms_np[12][1] * img.shape[0]
                        four_tip = handLms_np[16][1] * img.shape[0]
                        five_tip = handLms_np[20][1] * img.shape[0]

                        if is_close(one_tip, two_tip):
                            open_close=1
                        
                        elif is_close(one_tip, mid_tip):
                            open_close=2

                        elif is_close(one_tip, four_tip):
                            open_close=3

                        elif is_close(one_tip, five_tip):
                            open_close=4

                        
                        #回到音樂的起點
                        if where_music_song==len(music_song):
                                where_music_song=0

                        if not is_close(one_tip, two_tip) and open_close==1:
                            num_points+=1
                            note=whitch_note(music_song,where_music_song)
                            threading.Thread(target=play_music, args=(note,)).start()
                            where_music_song+=1
                            open_close=0
                        
                        elif not is_close(one_tip, mid_tip)and open_close==2:
                            num_points+=1
                            note=whitch_note(music_song,where_music_song)
                            threading.Thread(target=play_music, args=(note,)).start()
                            where_music_song+=1
                            open_close=0
                        elif not is_close(one_tip, four_tip)and open_close==3:
                            num_points+=1
                            note=whitch_note(music_song,where_music_song)
                            threading.Thread(target=play_music, args=(note,)).start()
                            where_music_song+=1
                            open_close=0

                        elif not is_close(one_tip, five_tip) and open_close==4:
                            num_points+=1
                            note=whitch_note(music_song,where_music_song)
                            threading.Thread(target=play_music, args=(note,)).start()
                            where_music_song+=1
                            open_close=0
                    



            # 將線加到影像上
            #cv2.line(img, (0, line_y), (img.shape[1], line_y), line_color, line_thickness)  # 繪製線
            cv2.line(img, (0, line_y), (img.shape[1], line_y), line_color, line_thickness)
            cv2.line(img,(210,0),(210,100),line_color,line_thickness)
            cv2.line(img,(425,0),(425,100),line_color,line_thickness)
            #score
            cv2.putText(img, f"Scores: {num_points}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            if cv2.waitKey(1) & 0xFF == ord('1'):
                part=1
                where_music_song=0
                
            if cv2.waitKey(1) & 0xFF == ord('2'):
                part=2
                where_music_song=0
                part_2=0
                
            if cv2.waitKey(1) & 0xFF == ord('3'):
                part=3
                where_music_song=0
            
            call_part(img,part)

            # 繪製手部關節點
            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)

            # 顯示影像
            cv2.imshow('img', img)
        

        # 偵測 'q' 鍵是否按下
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放攝影機
    cap.release()

    # 關閉所有視窗
    cv2.destroyAllWindows()


# 初始化pygame
pygame.mixer.init()

# 設定擷取影像的裝置
cap = cv2.VideoCapture(0)

# 設定 MediaPipe 手部偵測的參數
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)  # 只偵測一隻手
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 0), thickness=5)
handConStyle = mpDraw.DrawingSpec(color=(255, 255, 255), thickness=2)

# 設定線的初始狀態與域值的高低
line_y = 100
line_color = (255, 255, 255)  # 白色
line_thickness = 2

# 設定視窗大小
window_width = 800
window_height = 600



# 創建一個空白影像作為視窗
window = np.zeros((window_height, window_width, 3), dtype=np.uint8)

# 讀取圖片並調整大小以適應視窗
image_path = "test.jpg"  # 替換為您想要添加的圖片路徑
image = cv2.imread(image_path)
resized_image = cv2.resize(image, (window_width - 50, window_height - 50))  # 調整圖片大小以適應視窗
image_height, image_width, _ = resized_image.shape

# 在視窗上加載圖片
window[25:25 + image_height, 25:25 + image_width] = resized_image  # 在指定位置顯示圖片

# 顯示視窗
cv2.imshow("GUI Window", window)

# 設置鼠標事件回調函數
cv2.setMouseCallback("GUI Window", on_button_click)

# 等待左鍵按下
running = False
while not running:
    cv2.waitKey(1)

# 關閉視窗
cv2.destroyAllWindows()


# 跟踪手指狀態的字典，初始化為全部 False
finger_states = {4: False, 8: False, 12: False, 16: False, 20: False}


# 創建影像處理的執行緒
image_thread = threading.Thread(target=process_image)
image_thread.start()