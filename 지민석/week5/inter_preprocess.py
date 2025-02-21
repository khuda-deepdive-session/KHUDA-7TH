# 필요한 라이브러리 다시 로드
import pandas as pd
import numpy as np

# 파일 경로 설정
file_path = "dataset/ml-100k/ml-100k.inter"
output_path = "dataset/ml-100k/ml-100k_modified.inter"

# 데이터 불러오기 (TSV 형식, 공백 또는 탭으로 구분될 가능성 있음)
df = pd.read_csv(file_path, sep="\t")

# 'rating' 값을 0 또는 1로 랜덤하게 변경
df["rating:float"] = np.random.choice([0, 1], size=len(df))

# 변경된 데이터 저장 (파일 형식 유지)
df.to_csv(output_path, sep="\t", index=False)

# 저장 완료 메시지 출력
f"변경된 rating 값이 '{output_path}' 파일로 저장되었습니다."
