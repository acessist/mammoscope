# Verification report

Checked on Windows with Python 3.12 and TensorFlow 2.20.

- Eight data/integrity regression tests passed: input dimensions and label mapping, patient overlap rejection, duplicate file rejection, duplicate pixel rejection, invalid label rejection, missing class rejection, empty manifest rejection and invalid model hash rejection.
- The real-model desktop test passed: automatic loading of the selected regularized model, English window title, image upload, modality-confirmation gate, prediction with five normalized scores and JSON export.
- Exported regularized-head probabilities matched the fitted scikit-learn classifier on validation features within absolute tolerance 2e-5.
- Application and training sources compiled successfully.
- Published text files were checked for machine-specific user paths and development-tool branding.

Run `.venv/Scripts/python.exe -m unittest test_application -v`. The desktop check requires the local model weights, example images and a Windows desktop session; it is skipped when its model/examples are unavailable. The tests verify software behavior, not clinical validity. RESULTS.md contains actual image-level evaluation results.

Large weights and data remain local. The public repository contains source code, manifests, methods and evaluation metadata. A new checkout needs dataset preparation and training or a separately supplied trusted model bundle before inference.
