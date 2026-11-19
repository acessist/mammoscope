# MammoScope

MammoScope is a student project about using deep learning to classify mammogram images. The idea is simple: an image is loaded into a desktop app, and the model predicts one of five BI-RADS categories.

This repository shares the code and experiment results so people can understand the project. It is not a ready-to-install release. The training images and trained model files are not included.

## What the app does

The app shows the uploaded mammogram, a predicted BI-RADS category from 1 to 5, and a score for each category. It can also save the result as a JSON file. Images are processed on the computer.

BI-RADS categories describe a breast imaging assessment. They are not cancer stages. The app is trying to predict these image labels, rather than simply answer “cancer” or “no cancer.”

**This is a learning project, not a medical tool. It has not been clinically validated, and its scores are not cancer probabilities.**

## How it works

1. The image is converted to grayscale, resized to 224 × 224 pixels, and scaled to values between 0 and 1.
2. A neural network extracts useful patterns from the image.
3. A classifier uses these patterns to give scores for the five categories.
4. The app displays the category with the highest score.

Three approaches were compared:

- **A small CNN:** a network with three convolutional layers, trained from scratch.
- **ResNet50 with a dense classifier:** an existing network pretrained on ImageNet extracts image features. Its backbone stays fixed while a new classifier is trained.
- **ResNet50 with a regularized classifier:** the same features are used with a simpler classifier that limits how strongly it fits the training data. This became the selected model.

The project uses Python, TensorFlow/Keras, OpenCV, scikit-learn and Tkinter.

## The data

The images come from **CDD-CESM**, a public mammography dataset. This project uses its low-energy DM images and expert-provided BI-RADS labels.

After excluding images with multiple BI-RADS labels and labels outside 1–5, the experiment contains **931 images from 316 patients**.

| Part | Images | Patients | Purpose |
|---|---:|---:|---|
| Training | 558 | 190 | Learn from examples |
| Validation | 186 | 63 | Compare model settings |
| Test | 187 | 63 | Measure predictions after selection |

Images from the same patient stay in the same part. This helps avoid testing on another image of a patient already seen during training. The split stayed the same across the experiments.

## Results

| Model | Test accuracy |
|---|---:|
| Small CNN | 25.13% |
| ResNet50 with a dense classifier | 33.69% |
| ResNet50 with a regularized classifier | **47.59%** |

The selected model got **89 out of 187 test images correct**. For comparison, always guessing the most common training category would give 39.57% accuracy.

There was an improvement, but the results are still limited. Performance differs a lot between categories: the selected model did not correctly classify any BI-RADS 3 test images. Its balanced accuracy, which gives each category equal importance, is 33.58%.

Earlier test results were visible during later development, so this is an exploratory experiment. It has not been tested on a separate dataset from another source. The main takeaway is that making an app work and making its predictions reliable are two different challenges.

The detailed scores are in [metrics.json](models/cdd_cesm_regularized/metrics.json).

## What the files do

| File or folder | Role |
|---|---|
| `app.py` | Desktop interface, image upload and result display |
| `core.py` | Shared image processing, model loading and prediction |
| `train.py` | Training the small CNN |
| `train_resnet.py` | Training the first ResNet50 classifier |
| `improve_model.py` | Comparing the regularized classifiers |
| `prepare_cdd.py` | Preparing labels and patient-based data splits |
| `data/` | Image list, split information and preparation records |
| `models/` | Saved experiment settings and results; model weights are excluded |
| `test_application.py` | Checks for data handling and app behavior |
| `requirements.txt`, `requirements-lock.txt` | Python dependencies and recorded versions |
| `SETUP.ps1`, `START.cmd` | Setup and launch scripts used for the local application |

## Dataset credits

CDD-CESM is published through [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/cdd-cesm/) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Images were obtained from a [Kaggle mirror](https://www.kaggle.com/datasets/hanaraafatt/cdd-cesm/versions/1), and the labels came from [TCIA's annotation spreadsheet](https://www.cancerimagingarchive.net/wp-content/uploads/Radiology-manual-annotations.xlsx). The project selects a subset, removes spaces from filenames, and applies the image processing described above. The data providers do not endorse this model.

- [Khaled et al. — CDD-CESM dataset (2021)](https://doi.org/10.7937/29kw-ae92)
- [Khaled et al. — Dataset paper (2022)](https://doi.org/10.1038/s41597-022-01238-0)
- [Clark et al. — The Cancer Imaging Archive (2013)](https://doi.org/10.1007/s10278-013-9622-7)
