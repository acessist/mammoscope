# Training data

The local CDD-CESM data and manifest are ready. See ../DATA.md for acquisition, attribution and preparation. Source images are not included in Git.

For another dataset, provide a UTF-8 CSV with path,birads,patient_id,split. Image paths are relative to the CSV. BI-RADS labels must be expert-assigned integers 1–5; do not derive them from benign/malign pathology labels. Splits are train, val and test. Each split needs every class, and no patient may occur in multiple splits. Use de-identified mammograms with permission for the intended use. Supported inputs are 8-bit PNG/JPEG/BMP. DICOM conversion is outside this application.
