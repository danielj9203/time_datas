import lstm_cls

def bike_rental_training():
    """
    datetime - hourly date + timestamp  
    season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
    holiday - whether the day is considered a holiday
    workingday - whether the day is neither a weekend nor holiday
    weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy 
    2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
    3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
    4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
    temp - temperature in Celsius
    atemp - "feels like" temperature in Celsius
    humidity - relative humidity
    windspeed - wind speed
    casual - number of non-registered user rentals initiated
    registered - number of registered user rentals initiated
    count - number of total rentals
    
    Columns 명	데이터 내용
    Datetime	시간 (YYYY-MM-DD 00:00:00)
    Season	봄(1) 여름(2) 가을(3) 겨울(4)
    Holiday	공휴일(1) 그외(0)
    Workingday	근무일(1) 그외(0)
    Weather	아주깨끗한날씨(1) 약간의 안개와 구름(2) 약간의 눈,비(3) 아주많은비와 우박(4)
    Temp	온도(섭씨로 주어짐)
    Atemp	체감온도(섭씨로 주어짐)
    Humidity	습도
    Windspeed	풍속
    Casual	비회원의 자전거 대여량
    Registered	회원의 자전거 대여량
    Count	총 자전거 대여량 (비회원+회원)
    """
    bikes_train = lstm_cls.L('02_Bike Rental/02_bike_train.csv')
    bikes_train.loadNconvert_dataset()
    bikes_train.info()
    # 해당 칼럼들에 대한 그래프들 출력
    # bikes_train.show_dataset_infos(y='count', x5='Month', h5='season')
    # 사용되는 칼럼 지정
    bikes_train.scale_cols= ['season', 'holiday', 'weather', 'temp', 'humidity', 'windspeed', 'Year', 'Month', 'Day', 'count']
    bikes_train.normalize_dataset() # 정규화
    bikes_train.feature_cols= ['season', 'holiday', 'weather', 'temp', 'humidity', 'windspeed', 'Year', 'Month', 'Day'] # 상관성 칼럼
    bikes_train.label_cols= ['count'] # 예측할 칼럼
    bikes_train.create_train_dataset() # Train 데이터 셋 생성
    bikes_train.create_model('02_Bike Rental/model') # 학습 모델 생성
    bikes_train.train()

def bike_rental_test(): 
    bikes_test = lstm_cls.L('02_Bike Rental/02_bike_test.csv')
    bikes_test.info() # 변환 전 데이터 셋 정보 출력
    bikes_test.loadNconvert_dataset() #데이터 셋 변환
    bikes_test.info() # 변환 후 데이터 셋 정보 출력
    # DATA_FILE2에 count 칼럼이 없기 때문에 추가해준다.
    bikes_test.dataset['count'] = 0

    print(bikes_test.dataset.head())

    bikes_test.scale_cols= ['season', 'holiday', 'weather', 'temp', 'humidity', 'windspeed', 'Year', 'Month', 'Day', 'count'] # 사용할 칼럼 선택
    bikes_test.normalize_dataset() # 정규화
    bikes_test.feature_cols= ['season', 'holiday', 'weather', 'temp', 'humidity', 'windspeed', 'Year', 'Month', 'Day'] # 상관성 칼럼
    bikes_test.label_cols= ['count'] # 예측할 칼럼
    bikes_test.create_test_dataset() # Test 데이터 셋 생성
    bikes_test.create_model('02_Bike Rental/model') # 학습 모델이 저장되는 폴더 지정
    bikes_test.load_weights('epoch_0009.h5') # 학습 모델 로드
    bikes_test.test() # 예측 그래프 출력


if __name__=="__main__":
    
    bike_rental_test()

    print('done')