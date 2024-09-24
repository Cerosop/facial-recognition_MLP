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


def preprocess_data(X_train, X_test, n_components=100):
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
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score

# mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000, random_state=42)
# mlp_classifier.fit(X_train_processed, y_train)


# y_pred = mlp_classifier.predict(X_test_processed)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy * 100}%")



# entropy
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=X_train_processed.shape[1:]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(40, activation='softmax')
])

def entropy_loss(y_true, y_pred):
    eps = 1e-15  
    entropy = -tf.reduce_sum(y_true * tf.math.log(y_pred + eps), axis=1)
    return tf.reduce_mean(entropy)

model.compile(optimizer='adam',
              loss=entropy_loss,
              metrics=['accuracy'])

model.fit(X_train_processed, tf.keras.utils.to_categorical(y_train, num_classes=40), epochs=120, batch_size=16, validation_split=0, verbose=1)

test_loss, test_acc = model.evaluate(X_test_processed, tf.keras.utils.to_categorical(y_test, num_classes=40))
print("Test accuracy:", test_acc)