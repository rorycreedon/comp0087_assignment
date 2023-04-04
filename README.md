# Reconceptualising Automatic Text Summarisation Evaluation: Evidence from Long Legal Texts
Statistical NLP Research Project [read our report](Report.pdf) [view our presentation](Presentation.pdf)

## Project Description
Legal documents, such as contracts, case law and legislation, are often very long by nature. Reading, understanding and conducting analysis of these long documents can be a time-consuming and challenging task. This makes legal documents an area ripe for the use of automatic text summarisation (ATS). Within the Natural Language Processing (NLP) literature, the use of ATS has largely been evaluated using metrics such as ROUGE, which compare summarised text to a reference summary produced by a human. However, there are two key problems with using reference-based summary metrics such as ROUGE - firstly, they are limited by the quality of the human summary and secondly, they do not capture how useful the summarised text is for a range of different human tasks. To combat this, we propose an alternative framework for evaluating the effectiveness of summarised text for inference. Our results indicate that the widely used neural network systems, specifically transformers, fail to outperform older graph-based summary techniques on inference tasks despit higher ROUGE scores.

## Project Structure
- `experiments/`: Contains the scripts to make predictions. Main files include:
  - `train.py`: Fine-tunes the inference model.
  - `grid_search_opt.py`: Optimises the inference model via grid search.
  - `test.py`: Tests the inference model.
- `summaries/`: Contains the scripts to generate summaries
- `plots/`: Contains plots used in the report.
- `download_hf_data.py`: Downloads the ECHR dataset.
- `NLP_Demo_Notebook.ipynb`: Provides examples on how to use the models and perform inference.
- `plots.ipynb`: Generates plots for results.
- `requirements.txt`: Includes required dependencies.
- `results.xlsx`: Contains inference results across all models.

## Installation
Install the dependencies listed in the requirements.txt file
```python
pip install -r requirements.txt

