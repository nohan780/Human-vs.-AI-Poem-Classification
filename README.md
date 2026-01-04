# Human-vs.-AI-Poem-Classification

This project explores the detection of machine-generated text in the domain of poetry. Using a dataset of human-written and AI-generated poems (from models like GPT-Turbo and Gemini), this notebook performs Exploratory Data Analysis (EDA), text preprocessing, and builds classification models to distinguish between the two sources.

## Project Structure

The workflow inside the notebook (`cse427-project.ipynb`) follows these main steps:

1.  **Data Loading & Merging**: 
    -   Combines multiple datasets (`human_ai_poems_v1.jsonl`, `gpt-turbo`, `gemini`) into a unified DataFrame.
2.  **Exploratory Data Analysis (EDA)**:
    -   Visualizes the distribution of labels (Human vs. Machine).
    -   Analyzes poem lengths (character and token counts).
    -   Generates **Word Clouds** to compare vocabulary usage between human and AI poets.
3.  **Preprocessing**:
    -   Normalizes whitespace and line breaks.
    -   Removes non-ASCII characters and noise.
    -   Generates stable IDs to remove duplicates.
4.  **Modeling**:
    -   *Baseline*: Implements statistical models like **Logistic Regression** and **Quadratic Discriminant Analysis (QDA)** using TF-IDF features.
    -   *Deep Learning*: Fine-tunes a Transformer-based model (e.g., BERT/RoBERTa using `torch`) for sequence classification.
5.  **Evaluation**:
    -   Evaluates performance using Classification Reports (Precision, Recall, F1-Score) and Confusion Matrices.

## Dataset

The project utilizes the **Human and Language Model Generated Poems** dataset. It contains paired samples of poetry to facilitate the study of AI text detection.

* **Kaggle Dataset Link**: [Human and Language Model Generated Poems](https://www.kaggle.com/datasets/armandszokoly/human-and-language-model-generated-poems)

## Requirements

The project requires a Python environment (Python 3.11+ recommended) with the following primary libraries:

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `torch` (PyTorch)
* `wordcloud`

To install the dependencies, you can run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch wordcloud
