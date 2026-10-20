# Dataset and reproducibility

Source: [TCIA CDD-CESM](https://www.cancerimagingarchive.net/collection/cdd-cesm/), version 1, distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Citations:
- Khaled et al. (2021), CDD-CESM [Dataset], [doi:10.7937/29kw-ae92](https://doi.org/10.7937/29kw-ae92).
- Khaled et al. (2022), Scientific Data, [doi:10.1038/s41597-022-01238-0](https://doi.org/10.1038/s41597-022-01238-0).
- Clark et al. (2013), The Cancer Imaging Archive, [doi:10.1007/s10278-013-9622-7](https://doi.org/10.1007/s10278-013-9622-7).

Annotations were obtained from [TCIA's spreadsheet](https://www.cancerimagingarchive.net/wp-content/uploads/Radiology-manual-annotations.xlsx). Images were downloaded on 19 October 2026 from the [public Kaggle mirror](https://www.kaggle.com/datasets/hanaraafatt/cdd-cesm/versions/1), which identifies TCIA as its source, because the primary image download required an Aspera client. All 1,003 DM images matched the MD5 list inside the downloaded archive. This is an archive self-consistency check, not an independent comparison with the primary TCIA archive. Archive and annotation SHA-256 hashes are in data/cdd_cesm/preparation_report.json.

Only low-energy DM images are used. Subtracted images are excluded. Sixty-eight multi-valued labels and four BI-RADS 6 records are excluded, leaving 931 images from 316 patients. Pathology labels are not substituted for BI-RADS labels. Filenames have spaces removed. Images are decoded as grayscale, resized to 224 by 224 and divided by 255 during model input preparation. Providers do not endorse this application.

Download the archive and annotation spreadsheet from the sources above. Then run:

```powershell
.venv/Scripts/python.exe prepare_cdd.py --archive PATH_TO_ARCHIVE.zip --annotations PATH_TO_ANNOTATIONS.xlsx
```

The script reads the spreadsheet's `all` sheet and extracts only matching image files; it does not run programs in the archive. The resulting data/manifest.csv has relative paths and fixed patient-disjoint splits. Existing image files should be removed or the script run in a clean project when preparing a different archive, because extraction preserves existing filenames.

The split is StratifiedGroupKFold(5, shuffle=True, random_state=42): fold 0 test, fold 1 validation and folds 2–4 training. Counts are 558/186/187 images and 190/63/63 patients. The split is not changed to optimize reported results. Raw images, spreadsheets and large models are omitted from Git. Retain these citations, the license and transformation description when distributing the data or derivatives.
