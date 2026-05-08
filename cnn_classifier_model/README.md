# CNN Classifier Model

This folder contains the final CNN code for the IIVP digit-recognition challenge.

The model uses three image views for each digit:

- raw grayscale image
- Sobel edge image
- local stroke-density image

The final architecture is `StrokeViewCNN` in `src/iivp_project/model.py`.
It is the raw/edge/density CNN selected because it matches the best Kaggle-scored run.

## Train

From the project root:

```bash
PYTHONPATH=cnn_classifier_model/src python -m iivp_project.train_final
```

This saves the best checkpoint to:

```text
models/final_cnn.pt
```

## Create Kaggle File

After training, create the Kaggle CSV:

```bash
PYTHONPATH=cnn_classifier_model/src python -m iivp_project.predict_final
```

The prediction script uses:

```text
models/final_cnn.pt
```

This writes:

```text
submissions/final_submission.csv
```

The CSV format is:

```text
Id,Category
2,0
5,0
6,0
```

## Submit To Kaggle

```bash
kaggle competitions submit -c iivp-2026-challenge -f submissions/final_submission.csv -m "final cnn model"
```
