import lstm_cls

def electricSetData(electric):
    """
    1.date: Date in format dd/mm/yyyy
    2.time: time in format hh:mm:ss
    3.global_active_power: household global minute-averaged active power (in kilowatt)
    4.global_reactive_power: household global minute-averaged reactive power (in kilowatt)
    5.voltage: minute-averaged voltage (in volt)
    6.global_intensity: household global minute-averaged current intensity (in ampere)
    7.sub_metering_1: energy sub-metering No. 1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered).
    8.sub_metering_2: energy sub-metering No. 2 (in watt-hour of active energy). It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light.
    9.sub_metering_3: energy sub-metering No. 3 (in watt-hour of active energy). It corresponds to an electric water-heater and an air-conditioner.
    """

    electric.info() # 데이터 셋 정보 출력
    # 날짜에 대한 데이터 Column만 확인
    electric.loadNconvert_dataset() # 데이터 셋 변환
    # electric.show_dataset_infos('Global_active_power') # 해당 칼럼에 대한 그래프들 출력

    null_cols_idx=[]
    for i in range(len(electric.dataset.columns)): # 각 칼럼의 데이터 로드
        if not electric.dataset.iloc[:, i].notnull().all():
            null_cols_idx.append(i)

    for idx in null_cols_idx:
        electric.dataset.iloc[:,idx] = electric.dataset.iloc[:,idx].fillna(electric.dataset.iloc[:,idx].mean())

    electric.info() # 데이터 셋 정보 출력
    electric.scale_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'] # 사용되는 칼럼 지정
    electric.normalize_dataset() # 정규화
    electric.feature_cols = ['Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'] # 상관성 칼럼
    electric.label_cols = ['Global_active_power'] # 예측할 칼럼

    electric.create_trainNtest_dataset() # 학습 Train/Valid 생성
    electric.create_model('04_Electricity used/model') # 학습 모델 생성
     
if __name__=="__main__":
    electric = lstm_cls.L('04_Electricity used/04_household_power_consumption_50000.txt', sep=';', na_values=['nan','?'], parse_dates={'Date Time':['Date','Time']})
    electric.TEST_SIZE = 500 # 4.2.1 참고
    electric.WINDOW_SIZE = 50 # 4.2.1 참고
    electricSetData(electric) # 데이터 구성

    # 1. 학습
    # electric.train()

    # 2. 예측
    electric.load_weights('epoch_0028.h5')
    electric.test() # 예측 그래프 출력
    electric.performance_evaluation() # 성능평가 결과출력
