from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Conv1D, MaxPooling1D, Dropout, BatchNormalization, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class ModelTrainer:
    def __init__(self, max_features: int, embedding_dim: int, embedding_matrix: np.ndarray, max_len: int):
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.max_len = max_len

    def create_model(self, learning_rate: float = 0.001) -> Sequential:
        model = Sequential()
        model.add(Embedding(self.max_features, self.embedding_dim, weights=[self.embedding_matrix], input_length=self.max_len, trainable=False))
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, model: Sequential, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, batch_size: int = 32, epochs: int = 50):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(class_weights))
        
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=batch_size, callbacks=[early_stopping], class_weight=class_weights)
        return history

    def plot_history(self, history):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend()

        plt.show()
