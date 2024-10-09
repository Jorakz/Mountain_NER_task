# Mountain_NER_task
## Overview

This project is designed to recognize mountain names in text using Named Entity Recognition (NER) techniques and neural networks. The main objective is to extract mountain names from unstructured textual data.

## Database

* dataset_sentence: Contains 800 sentences, each featuring a mountain name.
* dataset_mountain: Includes 90 unique mountain names that are found in the dataset_sentence file.
* dataset_redacted: Stores tokenized sentence data with corresponding labels.
  
Labels are formatted as:
* 0: 'O': Represents tokens outside of any mountain name.
* 1: 'B-LOG': Indicates the beginning of a mountain name.
* 2: 'I-LOG': Represents the continuation of a mountain name in the same entity.

## Model Training

BERT (Bidirectional Encoder Representations from Transformers) base model (uncased) is used for token classification.
The dataset was tokenized and labels were aligned with the tokens.
The model was fine-tuned on the annotated dataset to classify mountain names effectively.
(https://huggingface.co/google-bert/bert-base-uncased)
## Usage

* create_database.ipynb: Creates the dataset_redacted from dataset_sentence and dataset_mountain.
* inference.py: Runs the fine-tuned BERT model to identify mountain names from text.
* train.py: Trains the BERT model for mountain name classification.
* requirements.txt: Lists the required libraries for the project.
* link_to_model_weights: Provides a link to the fine-tuned model weights.

## Conclusion

This project demonstrates the extraction of mountain names from text using NER and neural networks. Future improvements could enhance the model’s accuracy and robustness.
