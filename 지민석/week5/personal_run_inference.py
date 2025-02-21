import argparse
import torch
import numpy as np
import pandas as pd

from recbole.quick_start import load_data_and_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, default='saved/model.pth', help='Path to the trained model')
    parser.add_argument('--user_ids', '-u', type=str, required=True, help='Comma-separated list of user IDs to get recommendations for')

    args, _ = parser.parse_known_args()

    # 모델 및 데이터셋 불러오기
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(args.model_path)

    # device 설정
    device = config.final_config_dict['device']

    # user, item id -> token 변환 array (문자열 변환)
    user_id2token = list(map(str, dataset.field2id_token['user_id'].tolist()))  # numpy.ndarray → list(str)
    item_id2token = list(map(str, dataset.field2id_token['item_id'].tolist()))  # numpy.ndarray → list(str)

    # user-item sparse matrix
    matrix = dataset.inter_matrix(form='csr')

    # 입력된 사용자 ID 가져오기 (문자열 변환 후 비교)
    input_user_ids = set(map(str, args.user_ids.split(',')))

    # test 데이터에 존재하는 사용자 ID 확인
    available_user_ids = set(user_id2token)
    valid_user_ids = input_user_ids.intersection(available_user_ids)

    if not valid_user_ids:
        print(f"Error: 입력한 사용자 ID {input_user_ids}가 test 데이터에 없습니다. 가능한 user_id: {sorted(available_user_ids)[:10]} ...")
        exit(1)

    # 추천 결과 저장을 위한 딕셔너리
    recommendations = {}

    model.eval()
    for data in test_data:
        interaction = data[0].to(device)
        score = model.full_sort_predict(interaction)

        rating_pred = score.cpu().data.numpy().copy()
        batch_user_index = interaction['user_id'].cpu().numpy()

        # 선택한 user_id만 처리 (문자열로 비교)
        selected_idx = [i for i, uid in enumerate(batch_user_index) if user_id2token[uid] in valid_user_ids]
        if not selected_idx:
            continue

        selected_users = batch_user_index[selected_idx]
        selected_scores = rating_pred[selected_idx]

        selected_scores[matrix[selected_users].toarray() > 0] = 0
        ind = np.argpartition(selected_scores, -10)[:, -10:]

        arr_ind = selected_scores[np.arange(len(selected_scores))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(selected_scores)), ::-1]

        batch_pred_list = ind[np.arange(len(selected_scores))[:, None], arr_ind_argsort]

        # 예측값 저장
        for user, pred_items in zip(selected_users, batch_pred_list):
            user_id = user_id2token[user]  # 변환된 user_id
            item_ids = [item_id2token[item] for item in pred_items]
            recommendations[user_id] = item_ids

    # 데이터 저장 (한 줄에 user_id, 추천된 item_id 리스트)
    if recommendations:
        dataframe = pd.DataFrame(recommendations.items(), columns=["user", "recommended_items"])
        dataframe["recommended_items"] = dataframe["recommended_items"].apply(lambda x: ",".join(map(str, x)))
        dataframe.to_csv("personal_submission.csv", index=False)
        print(f'Inference done! Recommendations saved for users: {valid_user_ids}')
    else:
        print("Error: 추천 결과가 생성되지 않았습니다. test 데이터 내 사용자 ID를 확인하세요.")
