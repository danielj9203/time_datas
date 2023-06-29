import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
import os

class L:
    scale_cols= []              # 각 예제에서 적용할 칼럼들을 받는 리스트
    feature_cols= []            # 유저가 선택한 상관성이 높은 칼럼들을 받는 리스트
    label_cols= []              # 예측할 칼럼을 받는 리스트
    TEST_SIZE = 150             # 설명 4.2.1 참고
    WINDOW_SIZE = 20            # 설명 4.2.1 참고

    def __init__(self, data_path, header=0, encoding='cp949', sep=None, na_values=None, parse_dates=None): 
        self.data_path = data_path          # 학습 데이터(.CSV)파일 경로 저장
        # 학습 데이터(.CSV) 파일 경로와 pandas 모듈로 학습 데이터 로드
        self.dataset = pd.read_csv(self.data_path, header=header, encoding=encoding, sep=sep, na_values=na_values, parse_dates=parse_dates)

    def loadNconvert_dataset(self):
        # 각 예제에서 시간에 해당하는 칼럼을 추출
        self.date_name = self.dataset.columns[0]
        # object 타입의 시간 데이터 셋을 datetime 타입으로 변환
        # self.dataset['Date'] = pd.to_datetime(self.dataset[self.date_name], format='%d.%m.%Y %H:%M:%S')
        self.dataset['Date'] = pd.to_datetime(self.dataset[self.date_name])
        # Year, Month, Day, Weekday 칼럼을 추가
        self.dataset['Year'] = self.dataset['Date'].dt.year
        self.dataset['Month'] = self.dataset['Date'].dt.month
        self.dataset['Weekday'] = self.dataset['Date'].dt.weekday
        self.dataset['Day'] = self.dataset['Date'].dt.day

        # 시간 데이터를 가진 학습자료, 시간 데이터가 없는 학습자료 구분
        # 시간 데이터가 존재하면 Hour 컬럼까지 추가
        if len(self.dataset[self.dataset['Date'].dt.hour==0]) != len(self.dataset):
            self.dataset['Hour'] = self.dataset['Date'].dt.hour

    def info(self):
        print(self.dataset.info())          # 로드한 학습 데이터(.CSV) 구성을 출력합니다.

    def show_dataset_infos(self, y, x1='Year', x2='Month', x3='Day', x4='Hour', x5='Year', h5='Month'):
        # 각각의 속성과 예측의 결과값으로 쓰이는 y값과의 관계 파악
        self.fig = plt.figure(figsize=[12,10]) # Figure 크기 지정

        # 연도별 시계열 데이터 그래프
        # 위에서 생성한 Figure의 1사분면에 Subplot을 생성
        self.fig.add_subplot(2,2,1)
        sns.barplot(x=x1,y=y,data=self.dataset.groupby(x1)[y].mean().reset_index(), palette='Paired')   # 그래프 그리기

        # 월별 시계열 데이터 그래프
        # 위에서 생성한 Figure의 2사분면에 Subplot을 생성
        self.fig.add_subplot(2,2,2)
        sns.barplot(x=x2,y=y,data=self.dataset.groupby(x2)[y].mean().reset_index(), palette='Paired')   # 그래프 그리기

        # 일별 시계열 데이터 그래프
        # 위에서 생성한 Figure의 3사분면에 Subplot을 생성
        self.fig.add_subplot(2,2,3)
        sns.barplot(x=x3,y=y,data=self.dataset.groupby(x3)[y].mean().reset_index(), palette='Paired')   # 그래프 그리기

        # 시간별 시계열 데이터 그래프
        # 시간이 존재하는 데이터 셋일 때 시행
        # 위에서 생성한 Figure의 4사분면에 Subplot을 생성
        if len(self.dataset[self.dataset['Date'].dt.hour== 0]) != len(self.dataset):
            self.fig.add_subplot(2,2,4)
            sns.barplot(x=x4,y=y,data=self.dataset.groupby(x4)[y].mean().reset_index(), palette='Paired') # 그래프 그리기

        # 연도와 그에 따른 월별 시계열 데이터 그래프
        self.fig = plt.figure(figsize=[12,10]) # 그래프 크기 지정
        sns.pointplot(x=x5,y=y,hue=h5,data=self.dataset.groupby([x5,h5])[y].mean().reset_index(), palette='tab10') # 그래프 그리기
        plt.xlabel(x5) # x축 레이블 추가
        plt.ylabel(y) # y축 레이블 추가

        # 칼럼들 사이 상관성을 나타내는 히트맵 그래프
        self.fig = plt.figure(figsize=[30,30]) # 히트맵 크기 지정
        sns.heatmap(self.dataset.corr(), annot=True, square=True, cmap='Reds')
        plt.show() # 히트맵 출력

    def normalize_dataset(self):
        self.normalized=False
        if not self.scale_cols: # 칼럼들이 존재하지 않으면 return
            return

        self.dataset.sort_index(ascending=False).reset_index(drop=True)
        # MinMaxScaler() Scaler 객체 생성. 최소값 최대값을 0 ~ 1로 스케일링
        self.scaler = MinMaxScaler()
        # 훈련 데이터 크기 및 범위 정규화
        self.dataset_scaled = self.scaler.fit_transform(self.dataset[self.scale_cols]) 
        # 정규화 된 데이터를 이용하여 데이터 프레임 생성
        self.dataset_scaled = pd.DataFrame(self.dataset_scaled)
        self.dataset_scaled.columns = self.scale_cols
        self.normalized = True

    def make_dataset(self, data, label):
        feature_list= []
        label_list= []
        for i in range(len(data) - self.WINDOW_SIZE):
            # 설명 4.2.1 참고
            feature_list.append(np.array(data.iloc[i:i+self.WINDOW_SIZE]))
            label_list.append(np.array(label.iloc[i+self.WINDOW_SIZE]))

        return np.array(feature_list), np.array(label_list)
    
    def create_train_dataset(self):
        # 정규화가 안 된 상태면 return
        if not self.normalized:
            return
        # fuature, label cols가 존재하지 않으면 return
        if not self.feature_cols or not self.label_cols: 
            return

        self.train_dataset = self.dataset_scaled
        # Train학습 데이터 파일의 상관성 칼럼들에 대한 feature 생성
        self.train_feature = self.train_dataset[self.feature_cols]
        # Train학습 데이터 파일의 예측할 칼럼에 대한 label 생성
        self.train_label = self.train_dataset[self.label_cols]
        # make_dataset() 함수로 Train/Test 데이터로 분류
        self.train_feature, self.train_label = self.make_dataset(self.train_feature, self.train_label)
        # 사이킷 런을 이용하여 Train/Valid 데이터를 분할
        self.x_train, self.x_valid, self.y_train, self.y_valid =  train_test_split(self.train_feature, self.train_label, test_size=0.2)

    def create_test_dataset(self):
        # 정규화가 안 된 상태면 return
        if not self.normalized: 
            return
        # fuature, label cols가 존재하지 않으면 return
        if not self.feature_cols or not self.label_cols:
            return
        self.test_dataset = self.dataset_scaled
        self.test_feature = self.test_dataset[self.feature_cols]
        self.test_label = self.test_dataset[self.label_cols]
        self.test_feature, self.test_label = self.make_dataset(self.test_feature, self.test_label) # 예측에 필요한 값과 예측할 값을 make_dataset()함수로 분할

    def create_trainNtest_dataset(self, only_train=False, test_dataset=None):
        # 정규화가 안 된 상태면 return
        if not self.normalized:
            return
        # fuature, label cols가 존재하지 않으면 return
        if not self.feature_cols or not self.label_cols:
            return

        # 삼성 주가 테스트에는 사용하지 않기 ***********************************
        # # 데이터셋 양 줄여서 사용하는 법 #1        
        # temp_dataset = self.dataset_scaled[:10000]
        # self.train_dataset = temp_dataset[:-self.TEST_SIZE]
        # self.test_dataset = temp_dataset[-self.TEST_SIZE:]
        # # 데이터셋 양 줄여서 사용하는 법 #2
        # self.train_dataset = self.dataset_scaled[:10000-self.TEST_SIZE]
        # self.test_dataset = self.dataset_scaled[10000-self.TEST_SIZE:10000]
        # 삼성 주가 테스트에는 사용하지 않기 ***********************************

        # 삼성 주가 때는 무조건 이거 쓰기 **************************************
        # 설명 4.2.1 참고
        self.train_dataset = self.dataset_scaled[:-self.TEST_SIZE]
        # 설명 4.2.1 참고
        self.test_dataset = self.dataset_scaled[-self.TEST_SIZE:]
        # 삼성 주가 때는 무조건 이거 쓰기 **************************************
        
        # Train학습 데이터 파일의 상관성 칼럼들에 대한 feature 생성
        self.train_feature = self.train_dataset[self.feature_cols]
        # Train학습 데이터 파일의 예측할 칼럼에 대한 label 생성
        self.train_label = self.train_dataset[self.label_cols]

        # 학습에 쓰일 데이터와 예측에 사용할 데이터를 생성
        self.train_feature, self.train_label = self.make_dataset(self.train_feature, self.train_label)
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.train_feature, self.train_label, test_size=0.2) # 학습 
        self.test_feature = self.test_dataset[self.feature_cols]
        self.test_label = self.test_dataset[self.label_cols]
        self.test_feature, self.test_label = self.make_dataset(self.test_feature, self.test_label) # 예측

    def create_model(self, model_path):
        self.model_path = model_path # 학습 결과물(모델) 경로
        self.model = Sequential() # 레이어 층을 선형으로 구성
        self.model.add(LSTM(16,input_shape=(self.WINDOW_SIZE,len(self.feature_cols)),activation='relu',return_sequences=False)) # LSTM 모델 구축
        self.model.add(Dense(1))
        # 모델을 기계가 이해할 수 있도록 컴파일
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def create_model_m2m(self, model_path):
        self.model_path = model_path # 학습 결과물(모델) 경로
        self.model = Sequential() # 레이어 층을 선형으로 구성
        self.model.add(LSTM(16,input_shape=(self.WINDOW_SIZE,len(self.feature_cols)),activation='relu',return_sequences=False)) # LSTM 모델 구축
        self.model.add(Dense(2)) # 출력층 추가
        # 모델을 기계가 이해할 수 있도록 컴파일
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train(self, epochs=200, batch_size=16):
        # EalryStopping, 학습 모델 명, 체크포인트 지정 및 학습
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        filename = os.path.join(self.model_path, 'epoch_{epoch:04d}.h5')
        checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        history = self.model.fit(self.x_train,  self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.x_valid, self.y_valid), callbacks=[early_stop, checkpoint])
        
        # 훈련 손실 그래프
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['train', 'val'], loc='upper right')
        plt.show()

    def load_weights(self, model_name):
        # 저장된 모델 로드
        self.model.load_weights(os.path.join(self.model_path, model_name))

    def test(self):
        # 훈련하고 나온 값들을 precdict() 함수를 사용하여 예측
        pred = self.model.predict(self.test_feature) 
        plt.figure(figsize=(12, 9)) # Figure 크기 지정
        print(len(self.test_label), type(self.test_label))

        plt.plot(self.test_label, label='actual') # 그래프에 참값을 그립니다.
        plt.plot(pred, label='prediction') # 그래프에 예측값을 그립니다.

        plt.legend() # 범례 표시
        plt.show() # Figure 출력

    def test_m2m(self):
        # 훈련하고 나온 값들을 precdict() 함수를 사용하여 예측
        pred = self.model.predict(self.test_feature) 
        plt.figure(figsize=(12, 9)) # Figure 크기 지정
        print(len(self.test_label), type(self.test_label))

        # plt.plot(self.test_label, label='actual') # 그래프에 참값을 그립니다.
        # plt.plot(pred, label='prediction') # 그래프에 예측값을 그립니다.

        plt.plot(self.test_label[:,0], label='close') # 그래프에 참값을 그립니다.
        plt.plot(self.test_label[:,1], label='adj close') # 그래프에 참값을 그립니다.
        plt.plot(pred[:,0], label='close prediction') # 그래프에 예측값을 그립니다.
        plt.plot(pred[:,1], label='adj close prediction') # 그래프에 예측값을 그립니다.
        plt.legend() # 범례 표시
        plt.show() # Figure 출력

    def performance_evaluation(self):
        pred = self.model.predict(self.test_feature) 
        mse = round(mean_squared_error(self.test_label, pred), 6)               # MSE
        rmse = round(np.sqrt(mse), 6)                                           # RMSE

        mae = mean_absolute_error(self.test_label, pred) 
        mae = round(mae, 6)                                                     # MAE
        mape = 0
        for i in range(len(self.test_label)):
            mape += abs((self.test_label[i] - pred[i]) / self.test_label[i])
        mape = mape * 100 / len(self.test_label)
        mape = round(mape[0], 6)                                                # MAPE
        print(f"MSE = {mse}, RMSE = {rmse}")
        print(f"MAE = {mae}, MAPE = {mape}")