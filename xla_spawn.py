import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 데이터 생성
alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
beta = np.array([0.1, 0.2, 0.3, 0.4])
accuracy = np.array([
    [80.2, 81.1, 83.5, 85.2, 86.3],
    [81.0, 82.5, 84.0, 85.8, 87.0],
    [82.2, 83.8, 85.5, 87.3, 88.5],
    [83.1, 84.5, 86.3, 88.0, 89.3]
])
import seaborn as sns

# 히트맵 Plot
plt.figure(figsize=(8, 6))
sns.heatmap(accuracy, annot=True, xticklabels=alpha, yticklabels=beta, cmap='coolwarm')
plt.xlabel('Alpha (α)')
plt.ylabel('Beta (β)')
plt.title('Accuracy Heatmap')
plt.show()

