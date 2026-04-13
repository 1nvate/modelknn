import kagglehub
import pandas as pd
import os
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

path = kagglehub.dataset_download("gyanashish/healthcare-diabetes")
filename = os.listdir(path)[0]
ds = pd.read_csv(os.path.join(path, filename))

X = ds.drop('Outcome', axis=1)
y = ds['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': range(3, 21),
    'metric': ['minkowski', 'euclidean', 'cosine']
}

grid_search = GridSearchCV(knn, param_grid, cv=kf, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Результаты")
print(f"Лучшие параметры: {grid_search.best_params_}")
print(f"Лучшая точность на кросс-валидации: {grid_search.best_score_:.4f}")

test_accuracy = grid_search.score(X_test, y_test)
print(f"Точность на отложенной тестовой выборке: {test_accuracy:.4f}")
