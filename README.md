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

This version is the one selected for the final code because it reproduces the Kaggle-best local CSV from the 100-epoch run.

Known result:

```text
local validation accuracy: 1.0000
public Kaggle score: 0.9986702
```

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

## Notes

- `data/` is the only dataset folder used by the repo.
- The final checkpoint is `models/final_cnn.pt`.
- Extra experiment checkpoints and submissions are ignored.
