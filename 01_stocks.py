# import sys, os
# path = os.path.abspath('..')
# sys.path.insert(0, path)
                
import lstm_cls

def stocksSetData(stocksData):
    """
    open 시가
    high 고가
    low 저가
    close 종가
    volume 거래량
    Adj Close 주식의 분할, 배당, 배분 등을 고려해 조정한 종가
    """
    stocksData.loadNconvert_dataset()                           # 데이터 셋 변환
    stocksData.info()                                           # 데이터 셋 정보 출력
    # stocksData.show_dataset_infos('Close')                      # 해당 칼럼에 대한 그래프들 출력
    # stocksData.scale_cols = ['Open', 'High', 'Low', 'Year', 'Close', 'Adj Close']    # 사용되는 칼럼 지정
    stocksData.scale_cols = ['Open', 'High', 'Low', 'Close']    # 사용되는 칼럼 지정
    stocksData.normalize_dataset()                              # 정규화
    # stocksData.feature_cols = ['Open', 'High', 'Low', 'Year']           # 상관성 칼럼
    stocksData.feature_cols = ['Open', 'High', 'Low']           # 상관성 칼럼
    # stocksData.label_cols = ['Close', 'Adj Close']                           # 예측할 칼럼
    stocksData.label_cols = ['Close']                           # 예측할 칼럼
    stocksData.create_trainNtest_dataset()                      # 학습 Train/Valid 데이터 생성
    stocksData.create_model('01_Stocks/model')                  # 학습 모델 생성

if __name__=="__main__":
    stocksData = lstm_cls.L('01_Stocks/01_stocks_samsung.csv')  # 학습에 사용할 데이터 로드
    stocksData.TEST_SIZE = 150
    stocksData.WINDOW_SIZE = 20
    stocksSetData(stocksData)                                   # 데이터 구성

    # 1. 학습
    # stocksData.train(batch_size=16)
 
    # 2. 예측
    stocksData.load_weights('epoch_0024.h5')                    # 학습 모델 로드
    stocksData.test()                                           # 예측 그래프 출력
    stocksData.performance_evaluation()                         # 성능평가 결과출력