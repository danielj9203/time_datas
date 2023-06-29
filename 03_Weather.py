import lstm_cls

def weatherSetData(weather):
    """
    p (mbar) = 내부 압력을 정량화하는 데 사용되는 파스칼 SI 유도 압력 단위
    T (degC) = 섭씨 온도를 나타냅니다.
    VPmax (mbar) = 포화증기압을 나타냅니다.
    sh (g/kg) = 특정 습도, 특정습도는 습한 공기의 샘플에서 수증기의 질량을 샘플의 질량으로 나눈 것값을 나타냅니다.
    wv (m/s) = 풍속을 나타냅니다.
    """
    weather.info() # 정규화 전 데이터 셋 정보 출력
    weather.loadNconvert_dataset() # 데이터 셋 변환
    weather.info() # 정규화 후 데이터 셋 정보 출력
    # weather.show_dataset_infos('T (degC)') # 해당 칼럼에 대한 그래프들 출력
    weather.scale_cols = ['p (mbar)', 'T (degC)', 'VPmax (mbar)', 'sh (g/kg)', 'wv (m/s)'] # 사용되는 칼럼 지정
    weather.normalize_dataset() # 정규화
    weather.feature_cols = ['p (mbar)', 'VPmax (mbar)', 'sh (g/kg)', 'wv (m/s)'] # 상관성 칼럼
    weather.label_cols = ['T (degC)'] # 예측할 칼럼
    weather.create_trainNtest_dataset() # 학습 Train/Valid 데이터 생성
    weather.create_model('03_Weather Data/model') # 학습 모델 생성


if __name__=="__main__":
    weather = lstm_cls.L('03_Weather Data/03_weather.csv')

    weatherSetData(weather) # 데이터 구성

    # 1. 학습
    # weather.train()

    # # 2. 예측
    weather.load_weights('epoch_0013.h5') # 학습 모델 로드
    weather.test() # 예측 그래프 출력
    weather.performance_evaluation() # 성능평가 결과출력