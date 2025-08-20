import numpy as np
import pandas as pd

def calc_monthly_annual(df, factor=None):
    '''
    주어진 데이터프레임에 대해 핵심 지표를 추출함.
    데이터프레임 컬럼에는 factor(object), 계약월(datetime), 거래금액(int), 평단가(int)가 필요함
    주어진 factor에 대해, 월별 거래건수/평균거래금액/평균평단가/수익률, 연간 수익률/변동성, 그리고 12개월 이동평균과 이동표준편차를 구하는 함수
    '''
    if factor:
        
        ## 1. 월간 통계
        ### 월별 거래건수, 평균거래금액, 평균평단가
        monthly_stats = df.groupby([factor, '계약월']).agg({
            '거래금액': ['count', 'mean', 'median'],
            '평단가': 'mean'
        }).reset_index()
        monthly_stats.columns = [factor, '계약월', '월별거래건수', '월평균거래금액', '중위가격', '월평균평단가']

        ### 월별 수익률 계산
        monthly_stats.sort_values([factor,'계약월'], inplace=True)
        monthly_stats['월별수익률'] = monthly_stats.groupby(factor)['월평균거래금액'].transform(lambda x: np.log(x / x.shift(1)))

        ## 2. 연간 통계
        ### 연간수익률, 변동성
        monthly_stats_copy = monthly_stats.copy()
        monthly_stats_copy['연도'] = monthly_stats_copy['계약월'].dt.year
        annual_stats = (monthly_stats_copy
                        .groupby([factor, '연도'])
                        .agg(
                            연간수익률=('월별수익률', lambda x:np.prod(1 + x) - 1),
                            변동성=('월별수익률', 'std')
                            ).reset_index()
                        )

        ## 3. 추세분석
        ### 이동평균/이동표준편차
        monthly_stats['이동평균'] = monthly_stats.groupby(factor)['월평균거래금액'].transform(lambda x: x.rolling(12, min_periods=1).mean())
        monthly_stats['이동표준편차'] = monthly_stats.groupby(factor)['월평균거래금액'].transform(lambda x: x.rolling(12, min_periods=1).std())

    else:
        monthly_stats = df.groupby('계약월').agg({
            '거래금액': ['count', 'mean', 'median'],
            '평단가': 'mean'
        }).reset_index()
        monthly_stats.columns = ['계약월', '월별거래건수', '월평균거래금액', '중위가격', '월평균평단가']
        
        monthly_stats.sort_values('계약월', inplace=True)
        monthly_stats['월별수익률'] = np.log(monthly_stats['월평균거래금액'] / monthly_stats['월평균거래금액'].shift(1))

        ## 2. 연간 통계
        ### 연간수익률, 변동성
        monthly_stats_copy = monthly_stats.copy()
        monthly_stats_copy['연도'] = monthly_stats_copy['계약월'].dt.year
        annual_stats = (monthly_stats_copy
                        .groupby('연도')
                        .agg(
                            연간수익률=('월별수익률', lambda x: np.prod(1 + x) - 1),
                            변동성=('월별수익률', 'std')
                            ).reset_index()
                        )

        ## 3. 추세분석
        ### 이동평균/이동표준편차
        monthly_stats['이동평균'] = monthly_stats['월평균거래금액'].rolling(12, min_periods=1).mean()
        monthly_stats['이동표준편차'] = monthly_stats['월평균거래금액'].rolling(12, min_periods=1).std()
        
    return monthly_stats, annual_stats