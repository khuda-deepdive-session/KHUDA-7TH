import numpy as np
import pandas as pd

# 파일 경로 설정
file_path = "dataset/ml-100k/ml-100k.item"
output_path = "dataset/ml-100k/ml-100k_modified.item"

# 데이터 불러오기 (TSV 형식)
df = pd.read_csv(file_path, sep="\t")

# 'difficulty' 값을 0, 1, 2에서 랜덤하게 변경
df["difficulty:float"] = np.random.choice(["상","중","하"], size=len(df))


# 변경된 데이터 저장 (파일 형식 유지)
df.to_csv(output_path, sep="\t", index=False)

# 저장 완료 메시지 출력
f"변경된 데이터가 '{output_path}' 파일로 저장되었습니다."
