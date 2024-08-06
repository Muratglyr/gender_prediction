import numpy as np
from numpy import ndarray
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

def group_history(reader_history: DataFrame) -> dict:
    grouped = {}
    for _, user_id, page_id in reader_history.to_records():
        if user_id not in grouped:
            grouped[user_id] = []
        grouped[user_id].append(page_id)
    return grouped

def decode_embeddings(page_embeddings: DataFrame) -> Dict[str, ndarray]:
    return {
        page_id: np.frombuffer(bytes.fromhex(page_emb), dtype="f4") for _, page_id, page_emb in page_embeddings.to_records()
    }

def prepare_data(label_data: DataFrame, reader_history: DataFrame, page_embeddings: DataFrame, max_len: int, max_features: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, ndarray]]:
    user_id_to_label = {user_id: label for _, user_id, label in label_data.to_records()}
    user_id_to_history = group_history(reader_history)
    page_id_to_embedding = decode_embeddings(page_embeddings)

    X, y = [], []
    for user_id, pages in user_id_to_history.items():
        if user_id in user_id_to_label:
            filtered_pages = [int(page_id) if int(page_id) < max_features else 0 for page_id in pages]
            X.append(filtered_pages)
            y.append(user_id_to_label[user_id])

    X = pad_sequences(X, maxlen=max_len, padding='post')
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y, page_id_to_embedding
