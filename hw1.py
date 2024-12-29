import numpy as np

# (i1_points, i2_points) : 대응점 쌍
i1_points = np.array([
    [10, 37],
    [45, 98],
    [67, 25],
    [67, 75],
    [32, 61],
    [83, 17],
    [75, 84],
    [54, 84]
])
i2_points = np.array([
    [14, 28],
    [49, 89],
    [71, 16],
    [94, 66],
    [36, 52],
    [87, 8],
    [79, 39],
    [58, 75]
])

# RANSAC 파라미터 설정
num_iterations = 100       
threshold = 5.0           
best_inlier_count = 0      
best_translation = None    

np.random.seed(0)  
for _ in range(num_iterations):
    # 랜덤으로 점 2개를 뽑아서
    idx = np.random.choice(len(i1_points), 2, replace=False)
    sample_i1 = i1_points[idx]  # ex) [[x1, y1], [x2, y2]]
    sample_i2 = i2_points[idx]

    # Translation 추정 ----
    # 2개의 점이 있을 때, 실제로는 sample_i2 - sample_i1 각각을 구해
    # 두 벡터가 일치해야 이상적이지만, 작은 노이즈가 있다면 "평균"으로 추정
    translation = np.mean(sample_i2 - sample_i1, axis=0)

    # 추정된 Translation을 i1_points 전체에 적용 ----
    translated_i1 = i1_points + translation  # shape 동일 (N x 2)

    # 실제 i2_points와의 거리(오차) 계산 ----
    errors = np.linalg.norm(i2_points - translated_i1, axis=1)

    # threshold 이하 => inlier 개수 체크 ----
    inlier_count = np.sum(errors <= threshold)

    # 현재 모델(translation)이 best인지 갱신 ----
    if inlier_count > best_inlier_count:
        best_inlier_count = inlier_count
        best_translation = translation

# RANSAC 반복 끝난 후, best_translation이 최종 결과 -----
tx, ty = best_translation

# homogeneous 좌표계에서의 Translation 행렬 ----
translation_matrix = np.array([
    [1, 0, tx],
    [0, 1, ty],
    [0, 0, 1]
])

print(translation_matrix)


# PS C:\Users\cream> & C:/Users/cream/AppData/Local/Microsoft/WindowsApps/python3.11.exe c:/Users/cream/Desktop/hw1.py
# [[ 1.  0.  4.]
# [ 0.  1. -9.]
# [ 0.  0.  1.]]
# PS C:\Users\cream>