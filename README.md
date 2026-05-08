# IIVP Project - Group 26

Kaggle competition: https://www.kaggle.com/competitions/iivp-2026-challenge

This project is for digit recognition on the IIVP2026 Challenge dataset. The final adopted model is a small custom CNN that uses course-style image-processing views before classification.

## Final Model

The final model is:

```text
StrokeViewCNN
```

Model file:

```text
cnn_classifier_model/src/iivp_project/model.py
```

The CNN input has 3 channels:

- raw grayscale digit
- Sobel edge view
- local stroke-density view

This version is the one selected for the final code
## Project Structure

```text
cnn_classifier_model/
  README.md
  src/iivp_project/
    config.py
    data.py
    model.py
    train_final.py
    predict_final.py
    utils.py

data/
  train.csv
  test.csv
  sample_submission.csv
  train/
  test/

submissions/
  final_submission.csv

models/
  final_cnn.pt

eda.ipynb
eda_submissions.ipynb
report_visualizations.ipynb
```

## Run The Final CNN

Install the CNN requirements:

```bash
pip install -r cnn_classifier_model/requirements.txt
```

Train the final model:

```bash
PYTHONPATH=cnn_classifier_model/src python -m iivp_project.train_final
```

This saves the best checkpoint to:

```text
models/final_cnn.pt
```

Create the Kaggle CSV:

```bash
PYTHONPATH=cnn_classifier_model/src python -m iivp_project.predict_final
```

The generated file is:

```text
submissions/final_submission.csv
```

## Kaggle Submission

Submit the generated CSV:

```bash
kaggle competitions submit -c iivp-2026-challenge -f submissions/final_submission.csv -m "final cnn model"
```

The CSV format is:

```text
Id,Category
2,0
5,0
6,0
```

## Notebooks

The notebooks are for checking the work and preparing report/presentation visuals:

- `eda.ipynb`: explores the dataset, class balance, image quality, and sample digit shapes.
- `eda_submissions.ipynb`: compares submission CSV files and highlights disagreement cases.
- `report_visualizations.ipynb`: creates raw/edge/density views, model comparison tables, final submission checks, and classical HOG baseline visuals.

## Notes

- `data/` is the only dataset folder used by the repo.
- The final checkpoint is `models/final_cnn.pt`.
- Extra experiment checkpoints and submissions are ignored.
