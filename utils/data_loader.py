import yaml
from typing import Tuple, Dict
import pandas as pd

class DataLoader:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        label_data = pd.read_csv(self.config['data_paths']['label_data'])
        reader_history = pd.read_csv(self.config['data_paths']['reader_history'])
        page_embeddings = pd.read_csv(self.config['data_paths']['page_embeddings'])
        return label_data, reader_history, page_embeddings