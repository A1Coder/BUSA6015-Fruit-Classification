# Automated Fruit Quality Classification Using Convolutional Neural Networks

## Project Overview

This project was developed for BUSA6015 – Deep Learning at The University of the West Indies, Cave Hill Campus.

The study investigates the effectiveness of convolutional neural networks in classifying fresh and rotten apples, bananas and oranges, and evaluates the impact of regularisation on model generalisation.

## Dataset

Fruits Fresh and Rotten for Classification:

https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

The original dataset contains 13,599 images across six classes. Following the exclusion of augmented copies to address identified data leakage, 1,511 original images were retained.

## Environment

- Python 3.13
- TensorFlow 2.21.0
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Download and extract the Kaggle dataset into the project's `data` directory. Ensure that the directory contains the original `train` and `test` folders, each containing the six fruit categories.

## Execution Instructions

Run the following scripts sequentially:

```bash
python 01_eda.py
python 02_preprocessing.py
python 03_baseline_cnn.py
python 04_regularised_cnn.py
python 05_evaluation.py
```

## Final Results

| Metric | Baseline CNN | Regularised CNN |
|---|---:|---:|
| Test accuracy | 87.54% | 88.75% |
| Macro precision | 87.59% | 88.94% |
| Macro recall | 87.28% | 88.29% |
| Macro F1-score | 87.36% | 88.54% |
| Test loss | 0.4468 | 0.3156 |

The regularised CNN achieved the best observed test performance. When predictions were grouped into fresh and rotten categories, the model achieved 93.01% accuracy and detected 95.19% of rotten fruit images.

## Reproducibility

A random seed of 42 was used. Raw images, preprocessed arrays and trained model files are excluded from this repository to reduce storage requirements. These files can be regenerated using the dataset and scripts provided.

Minor numerical differences may occur across hardware and software environments.

Experimental results and visualisations are available in the `outputs` directory.