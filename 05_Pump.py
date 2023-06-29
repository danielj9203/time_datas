from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from keras.layers import LSTM

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

#region =============== 다시 제작 구역 (시작) =================
def df_masking(df, val1, val2):
    comp_col = 'machine_status'
    
    errstateDF = df[(df[comp_col] == val1) | (df[comp_col]==val2)].reset_index(drop=True)   # 고장 & 수리 전체
    errstateDF.to_csv('./05_Pump/full_ErrState.csv', index=False)
    
    # 위의 데이터 프레임에서 고장(2번)상태인 인덱스 구하기
    stt2_rows = errstateDF.index[errstateDF[comp_col] == val1]
    
    for i in range(7):
        if (i == 6):
            temp_csv = errstateDF.iloc[stt2_rows[i]:,:]
            print(" >> 행 개수 : ", len(temp_csv))
        else:
            temp_csv = errstateDF.iloc[stt2_rows[i]:stt2_rows[i+1], :]
            print(" >> 행 개수 : ", len(temp_csv))
            
        temp_csv.to_csv('./05_Pump/errstate_{0}.csv'.format(i), index = False)

def df_normal(df, val):
    comp_col =  'machine_status'
    normstateDF = df[(df[comp_col] == val)].reset_index(drop=True)   # 정상(1번)상태 전체
    normstateDF.to_csv('./05_Pump/full_NormState.csv', index=False)

def devidePUMP_TrainNTest():
    # errDF 로드
    errDF_list = []
    for i in range(7):
        errDF_list.append(pd.read_csv('./05_Pump/errstate_{}.csv'.format(i)))
    
    # ERR의 분리 : 0~4(train), 5~6(test)
    train_errDF = pd.concat([errDF_list[0],errDF_list[1], errDF_list[2],errDF_list[3],errDF_list[4]], ignore_index = True)
    test_errDF = pd.concat([errDF_list[5],errDF_list[6]], ignore_index = True)
    
    # normDF 로드
    normDF = pd.read_csv('./05_Pump/full_NormState.csv')
    
    # Norm의 분리
    tmp1_df = normDF.iloc[:122076,:]
    tmp2_df = normDF.iloc[122076:,:]
    train_normDF = tmp1_df.sample(frac=0.115) 
    test_normDF = tmp2_df.sample(frac=0.002)
    
    # Train, Test끼리 합체
    trainDF = pd.concat([train_normDF, train_errDF], ignore_index = True)
    testDF = pd.concat([test_normDF, test_errDF], ignore_index = True)
    
    # 시간순 정렬
    trainDF = trainDF.sort_values(by='Date', ascending=True)
    testDF = testDF.sort_values(by='Date', ascending=True)
    
    # 별도 엑셀 저장
    trainDF.to_csv('./05_Pump/trainDF.csv', index=False)
    testDF.to_csv('./05_Pump/testDF.csv', index=False)

def loadNconvert_dataset(df):
    # 각 예제에서 시간에 해당하는 칼럼을 추출
    date_name = df.columns[0]
    # object 타입의 시간 데이터 셋을 datetime 타입으로 변환
    df['Date'] = pd.to_datetime(df[date_name])
    # Year, Month, Day, Weekday 칼럼을 추가
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday
    df['Day'] = df['Date'].dt.day

    # 시간 데이터를 가진 학습자료, 시간 데이터가 없는 학습자료 구분
    # 시간 데이터가 존재하면 Hour 컬럼까지 추가
    if len(df[df['Date'].dt.hour==0]) != len(df):
        df['Hour'] = df['Date'].dt.hour

def normalize_dataset(df, scale_cols):
    # 정규화 방식 : StandardScaler
    scaler = StandardScaler()
    
    # 훈련 데이터 크기 및 범위 정규화
    df_scaled = scaler.fit_transform(df[scale_cols]) 
    
    # 정규화 된 데이터를 이용하여 데이터 프레임 생성
    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = scale_cols
    return df_scaled

# dataframe을 nparray형식으로 바꿈 → create_trainNvalid_dataset()에 사용됨.
def make_dataset(winSize, feature, label):
    feature_list = []
    label_list = []
    print(f'len(feature) {len(feature)}')
    for i in range(len(feature) - winSize):
        feature_list.append(np.array(feature.iloc[i:i+winSize]))
        label_list.append(np.array(label.iloc[i+winSize]))
    
    print("  >> make_dataset() → feature_list: {0}, label_list: {1}".format(np.shape(feature_list),np.shape(label_list)))
    return np.array(feature_list), np.array(label_list)

def create_trainNvalid_dataset(winSize, trainDF, testDF, feature_cols, label_cols, only_train=False, test_dataset=None):
    train_dataset = trainDF
    train_feature = train_dataset[feature_cols]
    print(f'feature_cols!! {feature_cols}')
    print(f'len(train_feature)!! {len(train_feature)}')
    train_label = train_dataset[label_cols]
    print("  >> train_feature : {}".format(train_feature))
    print("  >> len : {}".format(len(train_feature)))
    print("\n  >> train_label : {}".format(train_label))
    print("  >> len : {}".format(len(train_label)))
    
    train_feature, train_label = make_dataset(winSize, train_feature, train_label)
    print("  >> train_feature: {0}, train_label: {1}".format(train_feature.shape, train_label.shape))
    x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size = 0.2, random_state=1)
    
    test_dataset = testDF
    test_feature = test_dataset[feature_cols]
    test_label = test_dataset[label_cols]
    
    test_feature, test_label = make_dataset(winSize, test_feature, test_label)
    return x_train, x_valid, y_train, y_valid, train_feature, train_label, test_feature, test_label

def create_model(WINDOW_SIZE, feature_cols):
    # model_path : 학습 결과물(모델) 경로
    model = Sequential()  # 레이어 층을 선형으로 구성
    print(len(feature_cols))
    model.add(LSTM(16,input_shape=(WINDOW_SIZE, len(feature_cols)), activation='relu', return_sequences=False)) # LSTM 모델 구축
    model.add(Dense(1)) # 출력층 추가
    # 모델을 기계가 이해할 수 있도록 컴파일
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train(model_path, model, x_train, y_train, x_valid, y_valid, epochs=200, batch_size=16):
    # EalryStopping, 학습 모델 명, 체크포인트 지정 및 학습
    early_stop = EarlyStopping(monitor='val_loss', patience=20)
    filename = os.path.join(model_path, 'epoch_{epoch:04d}.h5')
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    history = model.fit(x_train, y_train,epochs=epochs,batch_size=batch_size,validation_data=(x_valid, y_valid),callbacks=[early_stop, checkpoint])

def test(model, test_feature, test_label):
    # 훈련하고 나온 값들을 precdict() 함수를 사용하여 예측
    pred = model.predict(test_feature) 
    plt.figure(figsize=(12, 9)) # Figure 크기 지정
    plt.plot(test_label, label='actual') # 그래프에 참값을 그립니다.
    plt.plot(pred, label='prediction') # 그래프에 예측값을 그립니다.
    plt.legend() # 범례 표시
    plt.show() # Figure 출력

def performance_evaluation(model, test_feature, test_label):
    pred = model.predict(test_feature) 
    mse = round(mean_squared_error(test_label, pred), 6) #MSE
    rmse = round(np.sqrt(mse), 6) #RMSE

    mae = mean_absolute_error(test_label, pred) 
    mae = round(mae, 6) #MAE
    mape = 0
    for i in range(len(test_label)):
        mape += abs((test_label[i] - pred[i]) / test_label[i])
    mape = mape * 100 / len(test_label)
    mape = round(mape[0], 6) #MAPE
    print(f"MSE = {mse}, RMSE = {rmse}")
    print(f"MAE = {mae}, MAPE = {mape}")

def info(df):
    print(df.info())        # 로드한 학습 데이터 구성을 출력

def show_dataset_infos(df, y, x1='Year', x2='Month', x3='Day', x4='Hour', x5='Year', h5='Month'):
    # 각각의 속성과 예측의 결과값으로 쓰이는 y값과의 관계 파악
    fig = plt.figure(figsize=[12,10]) # Figure 크기 지정

    # 연도별 시계열 데이터 그래프
    # 위에서 생선한 Figure의 1사분면에 Subplot을 생성
    fig.add_subplot(2,2,1)
    sns.barplot(x=x1,y=y,data=df.groupby(x1)[y].mean().reset_index(), palette='Paired') # 그래프 그리기

    # 월별 시계열 데이터 그래프
    # 위에서 생선한 Figure의 2사분면에 Subplot을 생성
    fig.add_subplot(2,2,2) 
    sns.barplot(x=x2,y=y,data=df.groupby(x2)[y].mean().reset_index(), palette='Paired') # 그래프 그리기

    # 일별 시계열 데이터 그래프
    # 위에서 생선한 Figure의 3사분면에 Subplot을 생성
    fig.add_subplot(2,2,3)
    sns.barplot(x=x3,y=y,data=df.groupby(x3)[y].mean().reset_index(), palette='Paired') # 그래프 그리기

    # 시간별 시계열 데이터 그래프
    # 시간이 존재하는 데이터 셋일 때 시행
    # 위에서 생선한 Figure의 4사분면에 Subplot을 생성
    if len(df[df['Date'].dt.hour== 0]) != len(df):
        fig.add_subplot(2,2,4)
        sns.barplot(x=x4,y=y,data=df.groupby(x4)[y].mean().reset_index(), palette='Paired') # 그래프 그리기

    # 연도와 그에 따른 월별 시계열 데이터 그래프
    fig = plt.figure(figsize=[12,10]) # 그래프 크기 지정
    sns.pointplot(x=x5,y=y,hue=h5,data=df.groupby([x5,h5])[y].mean().reset_index(), palette='tab10') # 그래프 그리기
    plt.xlabel(x5) # x축 레이블 추가
    plt.ylabel(y) # y축 레이블 추가

    # 칼럼들 사이 상관성을 나타내는 히트맵 그래프
    fig = plt.figure(figsize=[30,30]) # 히트맵 크기 지정
    sns.heatmap(df.corr(), annot=False, square=True, cmap='Reds')
    plt.show() # 히트맵 출력

#endregion=============== 다시 제작 구역 (끝) =================


if __name__ == "__main__":
    # region ======== 데이터 가공 및 제작 구역 ===========
    # pumpDF = pd.read_csv('./05_Pump/pump_sensor.csv') # 'machine_status'→ 1: 정상 / 2: 고장(순간) / 3: 수리중
    # pumpDF.info()
    # pumpDF.drop(pumpDF.columns[0], axis=1, inplace=True)
    # data_mapping = {'NORMAL' : 1, 'BROKEN' : 2, 'RECOVERING' : 3}
    # pumpDF['machine_status'] = pumpDF['machine_status'].map(data_mapping)
    
    # loadNconvert_dataset(pumpDF)    #시간정보 col 추가
    # show_dataset_infos(pumpDF, 'machine_status') # 데이터셋 정보 출력
    
    # # 'sensor_15'열 제거 & null값 0으로 변경
    # pumpDF = pumpDF.drop('sensor_15',axis=1)
    # pumpDF = pumpDF.fillna(0)
    
    # # 합치기
    # pumpDF = pumpDF.sort_values(by='Date', ascending=True)
    
    # # 상태에 따른 아예 별도로 분리하여 엑셀 저장
    # df_masking(pumpDF, 2, 3)
    # df_normal(pumpDF, 1)
    # devidePUMP_TrainNTest()       # 함수로 엑셀 분리 저장 및 데이터도 바로 반환
    # endregion ======== 데이터 가공 및 제작 구역 ===========
    
    # 데이터셋 가공 및 제작 이후 사용  (가공 이전에는 사용안해도 좋음.)
    trainDF = pd.read_csv('./05_Pump/trainDF.csv')
    testDF = pd.read_csv('./05_Pump/testDF.csv')

    # 칼럼 관련
    # scale_cols = ['sensor_00', 'sensor_01', 'sensor_02', 'sensor_03', 'sensor_04',
    #                 'sensor_05', 'sensor_06', 'sensor_07', 'sensor_08', 'sensor_09',
    #                 'sensor_10', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
    #                              'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19',
    #                 'sensor_20', 'sensor_21', 'sensor_22', 'sensor_23', 'sensor_24',
    #                 'sensor_25', 'sensor_26', 'sensor_27', 'sensor_28', 'sensor_29',
    #                 'sensor_30', 'sensor_31', 'sensor_32', 'sensor_33', 'sensor_34',
    #                 'sensor_35', 'sensor_36', 'sensor_37', 'sensor_38', 'sensor_39',
    #                 'sensor_40', 'sensor_41', 'sensor_42', 'sensor_43', 'sensor_44',
    #                 'sensor_45', 'sensor_46', 'sensor_47', 'sensor_48', 'sensor_49',
    #                 'sensor_50', 'sensor_51', 'Month', 'Day', 'Hour', 'machine_status']    # 사용되는 칼럼 지정
    # feature_cols = ['sensor_00', 'sensor_01', 'sensor_02', 'sensor_03', 'sensor_04',
    #                 'sensor_05', 'sensor_06', 'sensor_07', 'sensor_08', 'sensor_09',
    #                 'sensor_10', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
    #                              'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19',
    #                 'sensor_20', 'sensor_21', 'sensor_22', 'sensor_23', 'sensor_24',
    #                 'sensor_25', 'sensor_26', 'sensor_27', 'sensor_28', 'sensor_29',
    #                 'sensor_30', 'sensor_31', 'sensor_32', 'sensor_33', 'sensor_34',
    #                 'sensor_35', 'sensor_36', 'sensor_37', 'sensor_38', 'sensor_39',
    #                 'sensor_40', 'sensor_41', 'sensor_42', 'sensor_43', 'sensor_44',
    #                 'sensor_45', 'sensor_46', 'sensor_47', 'sensor_48', 'sensor_49',
    #                 'sensor_50', 'sensor_51', 'Month', 'Day', 'Hour']    # 입력 칼럼
    scale_cols = []    # 사용되는 칼럼 지정
    feature_cols = []    # 입력 칼럼
    for i in range(52):
        if i == 15: continue
        scale_cols.append(f'sensor_{i:02d}')
        feature_cols.append(f'sensor_{i:02d}')
    scale_cols.extend(['Month', 'Day', 'Hour', 'machine_status'])    # 사용되는 칼럼 지정
    feature_cols.extend(['Month', 'Day', 'Hour'])   # 입력 칼럼

    label_cols = ['machine_status']       # 예측 칼럼
    trainDF = normalize_dataset(trainDF, scale_cols)  # train 표준화
    testDF  = normalize_dataset(testDF,  scale_cols)  # test 표준화
    
    WinSize = 30
    # create_trainNvalid_dataset()으로 train에 필요한 각종 변수 지정
    x_train, x_valid, y_train, y_valid, train_feature, train_label, test_feature, test_label = create_trainNvalid_dataset(WinSize, trainDF, testDF, feature_cols, label_cols)
    
    # 모델 생성
    model_path = './05_Pump/model'
    model = create_model(WinSize, feature_cols)
    
    # 학습
    # train(model_path, model,x_train, y_train, x_valid, y_valid, epochs=500, batch_size = 128)
    
    # 예측
    model_name = 'epoch_0314.h5'
    model.load_weights(os.path.join(model_path, model_name))  # 학습 모델 로드
    test(model, test_feature, test_label)                     # 예측 그래프 출력
    # performance_evaluation(model, test_feature, test_label)   # 성능 평가 결과출력