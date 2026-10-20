# Manuscript methods and implementation appendix

Appending code alone does not resolve discrepancies in the original manuscript. Update its methods, findings and conclusions to match this implementation. The application outputs BI-RADS 1–5, not 0–6. Its data source is CDD-CESM, not the sources previously mentioned in the manuscript. The original three-convolution CNN is a comparison model; the ResNet50 transfer-learning and regularized-head experiments are extensions. Do not present the manuscript's seven-image example or its greater-than-90% claim as a result of these experiments. Exact measured results are in RESULTS.md.

## Data and partitioning

The low-energy DM subset of CDD-CESM v1 was used. Expert BI-RADS labels came from the `all` sheet of the TCIA annotation workbook. Images were acquired from a public mirror that credits TCIA. Its internal MD5 list matched all 1,003 DM images; equality with the primary TCIA archive was not independently authenticated. Dataset citations, license, acquisition details and transformations are in DATA.md; archive and annotation hashes are in the preparation report.

Sixty-eight multi-valued labels and four BI-RADS 6 labels were excluded, leaving 931 images from 316 patients. Pathology labels were not used as BI-RADS substitutes. Class values 1–5 map to training indices 0–4. StratifiedGroupKFold with five folds, shuffle enabled and seed 42 assigned fold 0 to test, fold 1 to validation and folds 2–4 to training. Counts were 558 training images from 190 patients, 186 validation images from 63 patients and 187 test images from 63 patients. No patients overlap between partitions. Exact files and equal preprocessed pixel arrays are checked for duplicates. This does not exclude all near-duplicates or source labeling errors.

## Shared image preprocessing

Pillow checks supported formats, pixel counts, 8-bit modes and file integrity. OpenCV decodes grayscale and resizes directly to 224 by 224 with linear interpolation, without preserving aspect ratio. Values are float32 divided by 255; inputs have shape (224,224,1). The same function is used in training and inference. No segmentation, augmentation, automatic modality detection or DICOM conversion is performed. Supported files are PNG/JPEG/BMP, at most 30 MB and 40 million pixels, with a minimum decoded dimension of 32 pixels.

## Baseline CNN

Three 3 by 3 valid Conv2D/ReLU layers with 32, 64 and 128 filters each precede 2 by 2 max pooling. Flatten is followed by Dense(128, ReLU) and Dense(5, softmax). The network has 11,169,029 parameters. It was trained from scratch for 10 epochs, batch size 8, Adam learning rate 0.001 and sparse categorical cross-entropy. Class weights were N/(5 times class count), computed only on the training partition. The epoch with the smallest validation loss was retained. TensorFlow deterministic operations and seed 42 were used. Exact numerical reproducibility across hardware/software is not guaranteed.

## Frozen ResNet50 and dense head

Keras ResNet50 uses ImageNet weights, include_top=False and global average pooling. Its backbone remains frozen. The grayscale channel is repeated three times, multiplied by 255 and centered by [103.939,116.779,123.68]. RGB-to-BGR reversal is immaterial because all three source channels are equal. These operations are stored inside the model. Features are 2,048 dimensional.

The first head uses Dense(128, ReLU), Dropout(0.5) and Dense(5, softmax), Adam learning rate 0.0003, batch size 16, 40 epochs and the same training-frequency class weights. Features are extracted once. The best epoch is selected by validation cross-entropy.

## Regularized softmax experiment

The existing frozen ResNet50 feature extractor is reused. StandardScaler is fit on training features only. Ten multinomial logistic regression heads are compared: C in {0.001, 0.01, 0.1, 1, 10}, each with either no class weighting or balanced weights. Scikit-learn LogisticRegression uses L2 regularization, its default lbfgs solver, max_iter=2000 and random_state=42. The minimum validation cross-entropy selects the configuration; validation accuracy is recorded but does not determine selection. Training and validation data are not merged for refitting. Test features are extracted only after this choice.

Standardization is folded into a Keras Dense(5, softmax) layer: kernel = coefficient-transpose / scale and bias = intercept minus mean multiplied by kernel. Predictions from the exported head and the fitted classifier are compared on validation features with absolute tolerance 2e-5. The full feature extractor and head are saved in one model.keras file. No fine-tuning of the convolutional backbone is performed. Selection records and actual results are retained.

## Evaluation and limitations

Report image-level accuracy, balanced accuracy, per-class precision/recall/F1, macro F1, confusion matrix and a majority-class baseline using the most frequent training label. The 187 test images correspond to 63 patients, not 187 independent patients. Earlier test results were visible before later experiments; this is exploratory development, not a fully blinded final evaluation. Model and epoch selection use validation loss, but a new independent evaluation set is needed for further claims. No external validation, confidence interval analysis or probability calibration has been performed. The system cannot establish clinical usefulness, and the 90% target is not a demonstrated result.

## Application and artifacts

The Tkinter interface performs work in a background thread and applies UI updates through a queue. It loads the default bundle named in models/active_model.json, checks schema, labels, preprocessing ID, SHA-256 and tensor dimensions, and requires user modality confirmation before prediction. Confirmation is not automatic image verification. The highest softmax score supplies the output class; all five scores are shown. JSON exports contain scores, model hash, test metrics and local timestamp. Images are not uploaded to a server.

Training creates separate output folders. model.keras contains inference weights; metadata.json records settings and model hash; metrics.json records evaluation; history.json and training_curves.png describe iterative runs; selection.json records the regularized search. Training audits use paths relative to the manifest. Metadata is written last so incomplete runs are not mistaken for complete bundles. The training tab supports the baseline CNN and dense-head ResNet50; the regularized experiment is reproduced with improve_model.py. All source files are included in CODE_APPENDIX.txt. Commands are in README.md.

References: DATA.md; Keras ResNet documentation at https://keras.io/api/applications/resnet/; scikit-learn LogisticRegression documentation at https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.
