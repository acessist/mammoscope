# Experiment results

The fixed patient-disjoint split contains 558 training, 186 validation and 187 test images (190, 63 and 63 patients). Selection uses validation cross-entropy. Earlier test results had already been inspected, so this is exploratory evaluation, not a fully blinded external test.

| Model | Validation loss | Test accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| cdd_cesm_cnn | 1.596915 | 25.13% | 19.83% | 0.1307 |
| cdd_cesm_resnet50 | 1.551795 | 33.69% | 29.64% | 0.2861 |
| cdd_cesm_regularized | 1.466622 | 47.59% | 33.58% | 0.3160 |

Default model: **cdd_cesm_regularized**. Its head uses C=0.001 and no class weighting. Validation accuracy is 40.86%. Test accuracy is 47.59% (89/187 correct). The 90% target was not reached.

The majority-class baseline predicts BI-RADS 1 for every image and achieves 39.57% accuracy. The selected model improves on this baseline in overall accuracy, but per-class performance remains poor. Scores are not calibrated cancer probabilities.

## Confusion matrix

Rows are expert reference labels; columns are predictions.

| Reference / predicted | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| 1 | 65 | 6 | 1 | 2 | 0 |
| 2 | 21 | 7 | 0 | 6 | 2 |
| 3 | 11 | 5 | 0 | 4 | 2 |
| 4 | 13 | 1 | 2 | 12 | 1 |
| 5 | 9 | 3 | 0 | 9 | 5 |

## Per-class results

| Class | Precision | Recall | F1 | Images |
|---|---:|---:|---:|---:|
| BI-RADS 1 | 0.5462 | 0.8784 | 0.6736 | 74 |
| BI-RADS 2 | 0.3182 | 0.1944 | 0.2414 | 36 |
| BI-RADS 3 | 0.0000 | 0.0000 | 0.0000 | 22 |
| BI-RADS 4 | 0.3636 | 0.4138 | 0.3871 | 29 |
| BI-RADS 5 | 0.5000 | 0.1923 | 0.2778 | 26 |

The original CNN selected epoch 2 of 10; the dense-head ResNet50 selected epoch 6 of 40. The regularized model compares ten configurations documented in models/cdd_cesm_regularized/selection.json. Preprocessing and patient partitions did not change. These experiments do not reproduce or establish the original manuscript's >90% claim. METHODS.md describes the implementation and its deviations.

No external validation or clinical validation has been performed. Multiple images from a patient are correlated. Further development requires a new independent evaluation set.
