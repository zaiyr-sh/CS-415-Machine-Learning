# Training, test and validation datasets

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

train_ratio = 0.80
test_ratio = 0.10
validation_ratio = 0.10

X, y = load_diabetes(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train,
    test_size = validation_ratio / (train_ratio + test_ratio)
)

print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)