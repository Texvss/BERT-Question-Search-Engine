# BERT Question Search Engine

## Description
Modular BERT-based system for question duplicate detection. Includes dataset prep, fine-tuning, validation, and top-k search.

## Installation
pip install -r requirements.txt

## Usage
python main.py  # Runs full pipeline: load, train, validate, infer.

## Modules
- dataset.py: Data loading/preprocessing
- model.py: Model/fine-tuning
- inference.py: Validation/top-k duplicates
- main.py: Entry point