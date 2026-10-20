"""Regression checks for data boundaries, model integrity and the desktop workflow."""
import csv
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from core import preprocess, read_manifest, load_bundle

ROOT=Path(__file__).resolve().parent

class DataChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        rng=np.random.default_rng(42)
        self.rows=[]
        for split in ('train','val','test'):
            for label in range(1,6):
                path=self.root/f'{split}_{label}.png'
                Image.fromarray(rng.integers(0,256,(48,64),dtype=np.uint8)).save(path)
                self.rows.append(dict(path=path.name,birads=label,patient_id=f'{split}_{label}',split=split))
    def tearDown(self):
        self.temp.cleanup()
    def manifest(self,rows):
        p=self.root/'manifest.csv'
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['path','birads','patient_id','split']);w.writeheader();w.writerows(rows)
        return p
    def test_shape_and_label_mapping(self):
        a=preprocess(self.root/self.rows[0]['path'])
        self.assertEqual(a.shape,(224,224,1));self.assertEqual(a.dtype,np.float32)
        self.assertTrue(0<=a.min()<=a.max()<=1)
        self.assertEqual([r['label'] for r in read_manifest(self.manifest(self.rows))[:5]],list(range(5)))
    def test_patient_overlap(self):
        self.rows[5]['patient_id']=self.rows[0]['patient_id']
        with self.assertRaisesRegex(ValueError,'patient occurs'):
            read_manifest(self.manifest(self.rows))
    def test_duplicate_file(self):
        with self.assertRaisesRegex(ValueError,'same file'):
            read_manifest(self.manifest(self.rows+[self.rows[0]]))
    def test_duplicate_pixels(self):
        (self.root/self.rows[5]['path']).write_bytes((self.root/self.rows[0]['path']).read_bytes())
        with self.assertRaisesRegex(ValueError,'Duplicate processed'):
            read_manifest(self.manifest(self.rows))
    def test_invalid_label(self):
        self.rows[0]['birads']=6
        with self.assertRaisesRegex(ValueError,'1–5'):
            read_manifest(self.manifest(self.rows))
    def test_missing_class(self):
        with self.assertRaisesRegex(ValueError,'Missing'):
            read_manifest(self.manifest(self.rows[:-1]))
    def test_empty_manifest(self):
        with self.assertRaisesRegex(ValueError,'CSV is empty'):
            read_manifest(self.manifest([]))
    def test_integrity_rejection(self):
        from core import LABELS,PREPROCESS
        (self.root/'model.keras').write_bytes(b'invalid')
        meta=dict(schema=1,labels=LABELS,preprocessing=PREPROCESS,trained=True,model_sha256='incorrect')
        (self.root/'metadata.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'integrity'):
            load_bundle(self.root)

class DesktopChecks(unittest.TestCase):
    @unittest.skipUnless((ROOT/'models/cdd_cesm_regularized/model.keras').is_file() and any((ROOT/'examples').glob('BIRADS_1_*.jpg')), 'Local model and examples required')
    def test_real_prediction_and_export(self):
        from app import App
        a=App();a.withdraw()
        def settle():
            deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                a.update()
                if not a.busy:return
                time.sleep(.03)
            self.fail('Background operation timed out')
        try:
            a.load_default();settle()
            self.assertIsNotNone(a.model)
            self.assertEqual(a.title(),'MammoScope | BI-RADS research application')
            image=next((ROOT/'examples').glob('BIRADS_1_*.jpg'))
            with patch('app.filedialog.askopenfilename',return_value=str(image)):
                a.choose_image()
            self.assertEqual(str(a.predict_button['state']),'disabled')
            a.confirmed.set(True);a.refresh();a.run_prediction();settle()
            self.assertIsNotNone(a.result)
            self.assertEqual(len(a.result['scores']),5)
            self.assertAlmostEqual(sum(a.result['scores'].values()),1,places=5)
            with tempfile.TemporaryDirectory() as temp:
                target=Path(temp)/'prediction.json'
                with patch('app.filedialog.asksaveasfilename',return_value=str(target)):
                    a.export_result()
                self.assertEqual(json.loads(target.read_text())['label'],a.result['label'])
        finally:
            a.destroy()

if __name__=='__main__':
    unittest.main(verbosity=2)
