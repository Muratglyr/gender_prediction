import yaml
import numpy as np
from sklearn.model_selection import train_test_split

from utils.data_loader import DataLoader
from utils.preprocess import prepare_data
from utils.model_trainer import ModelTrainer

class GenderPredictionPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.data_loader = DataLoader(config_path)
    
    def run(self):
        label_data, reader_history, page_embeddings = self.data_loader.load_data()

        max_len = self.config['model_params']['max_len']
        embedding_dim = self.config['model_params']['embedding_dim']
        max_page_id = max(int(page_id) for page_id in page_embeddings['page_id'])
        max_features = min(max_page_id + 1, 150001)

        X, y, page_id_to_embedding = prepare_data(label_data, reader_history, page_embeddings, max_len, max_features)
        embedding_matrix = np.zeros((max_features, embedding_dim))

        for page_id, embedding in page_id_to_embedding.items():
            index = int(page_id)
            if index < max_features:
                embedding_matrix[index] = embedding

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_trainer = ModelTrainer(max_features, embedding_dim, embedding_matrix, max_len)
        model = model_trainer.create_model(learning_rate=self.config['model_params']['learning_rate'])
        history = model_trainer.train_model(model, X_train, y_train, X_test, y_test, batch_size=self.config['training_params']['batch_size'], epochs=self.config['training_params']['epochs'])
        
        model_trainer.plot_history(history)

if __name__ == '__main__':
    pipeline = GenderPredictionPipeline(config_path='config.yaml')
    pipeline.run()
