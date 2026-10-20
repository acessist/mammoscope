"""Split expert-labeled CDD-CESM DM images at patient level.

python prepare_cdd.py --archive ../../work/cdd-cesm.zip --annotations data/cdd_cesm/annotations.xlsx
Only matching DM images are read; programs inside the archive are never executed.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import openpyxl
from sklearn.model_selection import StratifiedGroupKFold
from core import digest, preprocess, read_manifest


def prepare(archive, annotations):
    root = Path(__file__).resolve().parent
    target = root / 'data' / 'cdd_cesm'
    images = target / 'images'
    images.mkdir(parents=True, exist_ok=True)
    workbook = openpyxl.load_workbook(annotations, read_only=True, data_only=True)
    values = list(workbook['all'].values)
    source = [dict(zip(values[0], row)) for row in values[1:]]
    rows, excluded = [], []
    with zipfile.ZipFile(archive) as z:
        image_index = {}
        for name in z.namelist():
            if Path(name).suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            stem = Path(name).stem.replace(' ', '')
            if '_DM_' not in stem:
                continue
            if stem in image_index:
                raise ValueError('Duplicate filename in archive: ' + stem)
            image_index[stem] = name
        for row in source:
            if row['Type'] != 'DM':
                continue
            name, label = row['Image_name'].replace(' ', ''), row['BIRADS']
            # Do not arbitrarily reduce multiple findings to one label.
            if not isinstance(label, (int, float)) or label not in range(1, 6):
                excluded.append({'image':name, 'source_label':str(label), 'reason':'ambiguous_or_out_of_scope_label'})
                continue
            if name not in image_index:
                raise ValueError('Missing image: ' + name)
            member = image_index[name]
            path = images / (name + Path(member).suffix.lower())
            if not path.exists():
                path.write_bytes(z.read(member))
            pixels = preprocess(path)
            rows.append({'path':str(path.relative_to(root / 'data')).replace('\\','/'),
                         'birads':int(label), 'patient_id':str(row['Patient_ID']),
                         'pixel_hash':hashlib.sha256(pixels.tobytes()).hexdigest()})
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row['pixel_hash']].append(row)
    accepted = []
    for duplicates in by_hash.values():
        if len(duplicates) > 1:
            for row in duplicates:
                excluded.append({'image':Path(row['path']).stem, 'reason':'duplicate_processed_pixels'})
        else:
            accepted.extend(duplicates)
    accepted.sort(key=lambda r: r['path'])
    labels = np.asarray([r['birads'] for r in accepted])
    groups = np.asarray([r['patient_id'] for r in accepted])
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, indices) in enumerate(folds.split(np.zeros(len(accepted)), labels, groups)):
        split = 'test' if fold == 0 else ('val' if fold == 1 else 'train')
        for i in indices:
            accepted[i]['split'] = split
    manifest = root / 'data' / 'manifest.csv'
    with manifest.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path','birads','patient_id','split'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(accepted)
    read_manifest(manifest)
    report = {'dataset':'CDD-CESM v1', 'modality':'DM (low-energy CESM)',
              'source':'https://www.cancerimagingarchive.net/collection/cdd-cesm/',
              'image_download_mirror':'https://www.kaggle.com/datasets/hanaraafatt/cdd-cesm/versions/1',
              'annotations_source':'https://www.cancerimagingarchive.net/wp-content/uploads/Radiology-manual-annotations.xlsx',
              'license':'CC BY 4.0', 'access_date':'2026-10-19',
              'archive_sha256':digest(archive), 'annotations_sha256':digest(annotations),
              'split_method':'StratifiedGroupKFold(5, shuffle=True, random_state=42); fold0 test, fold1 val, folds2-4 train',
              'source_dm_count':sum(r['Type']=='DM' for r in source), 'used_count':len(accepted),
              'used_patients':len(set(groups)), 'excluded':excluded,
              'splits':{s:{'images':sum(r['split']==s for r in accepted),
                           'patients':len({r['patient_id'] for r in accepted if r['split']==s}),
                           'class_counts':dict(Counter(r['birads'] for r in accepted if r['split']==s))}
                        for s in ('train','val','test')}}
    (target/'preparation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='excluded'},ensure_ascii=False,indent=2))
    print('Excluded:',len(excluded))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--annotations', required=True)
    args = parser.parse_args()
    prepare(args.archive, args.annotations)
