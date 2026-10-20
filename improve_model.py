"""Validation-only selection of regularized softmax heads on frozen ResNet50 features."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
from core import LABELS, digest, load_bundle, read_manifest
from train_resnet import extract


def improve(manifest, source, output):
    from tensorflow import keras
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output folder must be empty.')
    output.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest)
    model, meta = load_bundle(source)
    if meta.get('architecture') != 'resnet50_imagenet_frozen_dense128_dropout05':
        raise ValueError('Source must be the frozen ResNet50 dense-head bundle.')
    inp = keras.Input((224, 224, 1), name="gray_normalized")
    feature_tensor = inp
    for layer in model.layers[1:-1]:
        feature_tensor = layer([feature_tensor] * 3) if isinstance(layer, keras.layers.Concatenate) else layer(feature_tensor, training=False)
    extractor = keras.Model(inp, feature_tensor)
    groups = {s: [r for r in rows if r['split'] == s] for s in ('train', 'val', 'test')}
    features = {s: extract(extractor, groups[s], s) for s in ('train', 'val')}
    labels = {s: np.array([r['label'] for r in groups[s]]) for s in groups}
    scaler = StandardScaler().fit(features['train'])
    x = {s: scaler.transform(features[s]) for s in features}
    candidates = []
    best = None
    for c in (0.001, 0.01, 0.1, 1.0, 10.0):
        for weight in (None, 'balanced'):
            head = LogisticRegression(C=c, class_weight=weight, max_iter=2000, random_state=42)
            head.fit(x['train'], labels['train'])
            probabilities = head.predict_proba(x['val'])
            record = {'C': c, 'class_weight': weight, 'validation_loss': float(log_loss(labels['val'], probabilities)),
                      'validation_accuracy': float(accuracy_score(labels['val'], probabilities.argmax(1)))}
            candidates.append(record)
            print(record, flush=True)
            if best is None or record['validation_loss'] < best[0]['validation_loss']:
                best = (record, head)
    record, head = best
    # Fold the fitted standardization into one Dense layer for portable inference.
    dense = keras.layers.Dense(5, activation='softmax', name='regularized_softmax')
    full = keras.Model(extractor.input, dense(extractor.output), name='resnet50_regularized')
    kernel = head.coef_.T / scaler.scale_[:, None]
    bias = head.intercept_ - scaler.mean_ @ kernel
    dense.set_weights([kernel.astype('float32'), bias.astype('float32')])
    actual = np.asarray(dense(features['val']))
    np.testing.assert_allclose(actual, head.predict_proba(x['val']), atol=2e-5)
    print('Saving model bundle...', flush=True)
    full.save(output / 'model.keras')
    print('Model saved.', flush=True)
    # The selected configuration is fixed before extracting/evaluating test features.
    test = extract(extractor, groups['test'], 'test')
    probabilities = np.asarray(dense(test))
    pred = probabilities.argmax(1)
    truth = labels['test']
    metrics = {'test_loss': float(log_loss(truth, probabilities)),
               'test_accuracy': float(accuracy_score(truth, pred)),
               'balanced_accuracy': float(balanced_accuracy_score(truth, pred)),
               'majority_baseline_accuracy': float(np.mean(truth == np.bincount(labels['train']).argmax())),
               'confusion_matrix': confusion_matrix(truth, pred, labels=list(range(5))).tolist(),
               'classification_report': classification_report(truth, pred, labels=list(range(5)), target_names=LABELS, output_dict=True, zero_division=0)}
    meta = {k:v for k,v in meta.items() if k not in ('epochs_requested','best_epoch','batch_size','optimizer','learning_rate','class_weights','training_seconds')}
    meta.update(architecture='resnet50_imagenet_frozen_regularized_softmax', selection='minimum validation cross-entropy across ten regularized heads',
                selected_configuration=record, candidates=candidates, metrics=metrics, model_sha256=digest(output/'model.keras'),
                manifest_sha256=digest(manifest), exploratory=True)
    for name, value in [('metrics.json',metrics),('selection.json',candidates),('metadata.json',meta)]:
        (output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='data/manifest.csv')
    p.add_argument('--source',default='models/cdd_cesm_resnet50')
    p.add_argument('--output',default='models/cdd_cesm_regularized')
    a=p.parse_args()
    improve(a.manifest,a.source,a.output)
