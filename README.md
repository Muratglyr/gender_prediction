# Demographic Prediction

This project aims to build a model that predicts a reader's gender based on their reading history and page embeddings.

## Installation

1. Clone the repository:
    git clone https://github.com/yourusername/demographic-prediction.git
    cd demographic-prediction   

2. Install the required packages:
    pip install -r requirements.txt


## Usage

1. Ensure the data files are in the `resources` directory:
- `challenge_data.csv`
- `challenge_reader_history.csv`
- `challenge_page_embeddings.csv`

2. Run the main script to train and evaluate the model:
    python main.py


## Project Structure

- `data_loader.py`: Contains the `DataLoader` class for loading data.
- `preprocess.py`: Contains preprocessing functions.
- `model_trainer.py`: Contains the `ModelTrainer` class for creating, training, and evaluating the model.
- `main.py`: The main script to orchestrate data loading, model training, and evaluation.
- `config.yaml`: Configuration file for input paths, model parameters, and training parameters.
- `resources/`: Directory containing the dataset files.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation.
