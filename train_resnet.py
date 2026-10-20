"""Frozen ImageNet ResNet50 features and a trained classifier on patient-disjoint data."""
import argparse
import json
import os
from pathlib import Path
import platform
import time
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import numpy as np
from core import LABELS, PREPROCESS, digest, preprocess, read_manifest


def train_resnet(manifest, output, epochs=40, log=print):
    if not 1 <= epochs <= 200:
        raise ValueError('Epochs must be between 1 and 200.')
    started=time.monotonic()
    rows=read_manifest(manifest)
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output folder must be empty.')
    output.mkdir(parents=True,exist_ok=True)
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import classification_report,confusion_matrix,balanced_accuracy_score
    keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    base=keras.applications.ResNet50(include_top=False,weights='imagenet',pooling='avg')
    base.trainable=False
    inp=keras.Input((224,224,1),name='gray_normalized')
    x=keras.layers.Concatenate()([inp,inp,inp])
    x=keras.layers.Rescaling(255.0)(x)
    # For grayscale R=G=B, reversing to BGR does not change the values.
    x=keras.layers.Normalization(mean=[103.939,116.779,123.68],variance=[1.,1.,1.])(x)
    features=base(x,training=False)
    extractor=keras.Model(inp,features,name='frozen_resnet50')
    groups={s:[r for r in rows if r['split']==s] for s in ('train','val','test')}
    feature_arrays={}
    for split in ('train','val'):
        feature_arrays[split]=extract(extractor,groups[split],split,log)
    head=keras.Sequential([keras.Input((2048,)),keras.layers.Dense(128,activation='relu'),
                           keras.layers.Dropout(0.5),keras.layers.Dense(5,activation='softmax')],name='birads_head')
    head.compile(optimizer=keras.optimizers.Adam(0.0003),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    labels={s:np.asarray([r['label'] for r in groups[s]]) for s in groups}
    counts=np.bincount(labels['train'],minlength=5)
    weights={i:float(len(labels['train'])/(5*counts[i])) for i in range(5)}
    class Progress(keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs=None):
            log(f'Epoch {epoch+1}/{epochs} | '+ ' | '.join(f'{k}: {v:.4f}' for k,v in logs.items()))
    history=head.fit(feature_arrays['train'],labels['train'],validation_data=(feature_arrays['val'],labels['val']),
                     epochs=epochs,batch_size=16,class_weight=weights,verbose=0,
                     callbacks=[Progress(),keras.callbacks.ModelCheckpoint(str(output/'head.keras'),monitor='val_loss',save_best_only=True)])
    head=keras.models.load_model(output/'head.keras',safe_mode=True)
    full=keras.Model(inp,head(features),name='resnet50_birads')
    full.save(output/'model.keras')
    feature_arrays['test']=extract(extractor,groups['test'],'test',log)
    loss,acc=head.evaluate(feature_arrays['test'],labels['test'],verbose=0)
    scores=head.predict(feature_arrays['test'],verbose=0)
    pred=scores.argmax(axis=1)
    metrics={'test_loss':float(loss),'test_accuracy':float(acc),
             'balanced_accuracy':float(balanced_accuracy_score(labels['test'],pred)),
             'majority_baseline_accuracy':float(np.mean(labels['test']==np.argmax(counts))),
             'confusion_matrix':confusion_matrix(labels['test'],pred,labels=list(range(5))).tolist(),
             'classification_report':classification_report(labels['test'],pred,labels=list(range(5)),target_names=LABELS,output_dict=True,zero_division=0)}
    hist={k:[float(v) for v in vs] for k,vs in history.history.items()}
    meta={'schema':1,'trained':True,'labels':LABELS,'preprocessing':PREPROCESS,
          'architecture':'resnet50_imagenet_frozen_dense128_dropout05','clinical_validated':False,
          'seed':42,'epochs_requested':epochs,'batch_size':16,'best_epoch':int(np.argmin(hist['val_loss']))+1,
          'selection':'minimum validation loss','optimizer':'Adam','learning_rate':0.0003,
          'class_weights':weights,'tensorflow':tf.__version__,'python':platform.python_version(),
          'platform':platform.platform(),'devices':[d.name for d in tf.config.list_physical_devices()],
          'training_seconds':time.monotonic()-started,'model_sha256':digest(output/'model.keras'),
          'manifest_sha256':digest(manifest),'metrics':metrics,
          'counts':{s:len(rs) for s,rs in groups.items()},
          'patient_counts':{s:len({r['patient_id'] for r in rs}) for s,rs in groups.items()}}
    for name,data in [('metrics.json',metrics),('history.json',hist),('data_audit.json',[{**r, 'path': Path(os.path.relpath(r['path'], Path(manifest).resolve().parent)).as_posix()} for r in rows])]:
        (output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric,title in zip(axes,['accuracy','loss'],['Accuracy','Loss']):
        ax.plot(range(1,epochs+1),hist[metric],label='Training')
        ax.plot(range(1,epochs+1),hist['val_'+metric],label='Validation')
        ax.set(title=title,xlabel='Epoch');ax.legend()
    fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=150);plt.close(fig)
    (output/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    log(json.dumps(metrics,indent=2))


def extract(model,rows,split,log=print):
    values=[]
    for start in range(0,len(rows),8):
        batch=np.stack([preprocess(r['path']) for r in rows[start:start+8]])
        values.append(np.asarray(model(batch,training=False)))
        if start%80==0: log(f'{split}: feature extraction {start}/{len(rows)}')
    return np.concatenate(values)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--epochs',type=int,default=40)
    args=parser.parse_args()
    train_resnet(args.manifest,args.output,args.epochs)
