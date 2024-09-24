import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def load_dataset(data_path):
    X = []
    y = []
    for i, folder in enumerate(os.listdir(data_path)):
        if 'Non' in folder:
            continue
        if i >= 40:
            break
        for file in os.listdir(os.path.join(data_path, folder)):
            if '.db' in file:
                continue
            img_path = os.path.join(data_path, folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            X.append(img.flatten())
            y.append(i)
    return np.array(X), np.array(y)

data_path = "ORL3232"
X, y = load_dataset(data_path)

X_train = np.array([x for i, x in enumerate(X) if i % 2 == 0])
X_test = np.array([x for i, x in enumerate(X) if i % 2 == 1])
y_train = np.array([x for i, x in enumerate(y) if i % 2 == 0])
y_test = np.array([x for i, x in enumerate(y) if i % 2 == 1])


def preprocess_data(X_train, X_test, n_components=9):
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    scaler = MinMaxScaler()
    scaler.fit(X_train_pca)
    X_train_normalized = scaler.transform(X_train_pca)
    X_test_normalized = scaler.transform(X_test_pca)
    
    return X_train_normalized, X_test_normalized

X_train_processed, X_test_processed = preprocess_data(X_train, X_test)


# cross entropy
import numpy as np
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

class MLPClassifierLM(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(20,), activation='relu', max_iter=1000, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.random_state = random_state

    def _initialize_weights(self, n_features, n_classes):
        rng = np.random.RandomState(self.random_state)
        n_hidden = self.hidden_layer_sizes[0]
        coef_ = rng.randn(n_features, n_hidden)
        intercept_ = np.zeros(n_hidden)
        self.coefs_ = [coef_]
        self.intercepts_ = [intercept_]

        if len(self.hidden_layer_sizes) > 1:
            for i in range(1, len(self.hidden_layer_sizes)):
                coef_i = rng.randn(n_hidden, self.hidden_layer_sizes[i])
                intercept_i = np.zeros(self.hidden_layer_sizes[i])
                self.coefs_.append(coef_i)
                self.intercepts_.append(intercept_i)

        self.coefs_.append(rng.randn(n_hidden, n_classes))
        self.intercepts_.append(np.zeros(n_classes))

    def _forward_pass(self, X):
        activations = X
        for i in range(self.n_layers_):
            activations = self._activation(activations, self.coefs_[i], self.intercepts_[i])
        return activations

    def _activation(self, X, coef, intercept):
        if self.activation == 'relu':
            return np.maximum(0, np.dot(X, coef) + intercept)
        elif self.activation == 'logistic':
            return 1.0 / (1.0 + np.exp(-(np.dot(X, coef) + intercept)))
        else:
            raise ValueError("Activation function '%s' not supported." % self.activation)

    def _loss(self, params, X, y):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_classes = self.classes_.shape[0]
        self._unpack_params(params, n_features, n_classes)

        activations = self._forward_pass(X)
        output_activation = activations

        eps = 1e-15
        loss = -np.sum(y * np.log(output_activation + eps)) / n_samples

        return loss

    def _unpack_params(self, params, n_features, n_classes):
        start = 0
        end = 0
        for i in range(self.n_layers_ - 1):
            end += (n_features + 1) * self.hidden_layer_sizes[i]
            self.coefs_[i] = np.reshape(params[start:end], (n_features + 1, self.hidden_layer_sizes[i]))[:-1]
            start = end
            n_features = self.hidden_layer_sizes[i]

        self.coefs_[-1] = np.reshape(params[start:], (n_features + 1, n_classes))[:-1]

    def fit(self, X, y):
        self.onehot_encoder_ = OneHotEncoder()
        y = self.onehot_encoder_.fit_transform(y.reshape(-1, 1)).toarray()
        self.classes_ = self.onehot_encoder_.categories_[0]
        self.n_layers_ = len(self.hidden_layer_sizes) + 1

        n_features = X.shape[1]
        n_classes = y.shape[1]

        self._initialize_weights(n_features, n_classes)

        params = np.hstack([coef.flatten() for coef in self.coefs_ + self.intercepts_])

        res = least_squares(self._loss, params, method='lm', args=(X, y), max_nfev=self.max_iter)

        self._unpack_params(res.x, n_features, n_classes)

        return self

    def predict_proba(self, X):
        return self._forward_pass(X)

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

# 使用 Levenberg-Marquardt (LM) 算法
mlp_classifier = MLPClassifierLM(hidden_layer_sizes=(1,), activation='relu', max_iter=1000, random_state=42)
mlp_classifier.fit(X_train_processed, y_train)

y_pred_tr = mlp_classifier.predict(X_train_processed)
accuracy = accuracy_score(y_train, y_pred_tr)
print(f"Training Accuracy: {accuracy * 100}%")
y_pred = mlp_classifier.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)
print(f"Testing Accuracy: {accuracy * 100}%")
