import pandas as pd
import numpy as np


# 파일 경로 설정
file_path = "dataset/ml-100k/ml-100k.user"
output_path = "dataset/ml-100k/ml-100k_modified.user"

# 데이터 불러오기 (TSV 형식)
df = pd.read_csv(file_path, sep="\t")

# 'occupation:token' 및 'zip_code:token' 열 삭제
df.drop(columns=["occupation:token", "zip_code:token"], inplace=True)

# 변경된 데이터 저장 (파일 형식 유지)
df.to_csv(output_path, sep="\t", index=False)

# 저장 완료 메시지 출력
f"변경된 데이터가 '{output_path}' 파일로 저장되었습니다."
