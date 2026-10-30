import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image

from classifier.common import file_hash, seed_everything
from classifier.webcam import crop_roi
from classifier.data import image_transform
from recognition.compare import ModelBank, PredictionWorker, PROJECT, render


class RecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(42)
        cls.bank = ModelBank(torch.device('cpu'))
        cls.hashes = {m['checkpoint']: file_hash(m['checkpoint']) for m in cls.bank.metadata.values()}
        cls.frame = np.random.default_rng(3).integers(0,256,(180,320,3),dtype=np.uint8)

    @classmethod
    def tearDownClass(cls):
        for path, digest in cls.hashes.items():
            if file_hash(path) != digest:
                raise AssertionError('A checkpoint changed during recognition testing')

    def test_original_matches_existing_webcam_preprocessing(self):
        roi,_=crop_roi(self.frame)
        image=Image.fromarray(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
        with torch.inference_mode():
            expected=self.bank.models['original'](image_transform(False)(image).unsqueeze(0)).softmax(1)[0]
        result=self.bank.predict(self.frame,'original')
        np.testing.assert_allclose(result['all_class_scores'],expected.numpy(),atol=1e-7)

    def test_detector_uses_exact_preprocessing_and_skips_other_heads(self):
        self.assertIn('detector',self.bank.models)
        from detection.data import letterbox,tensorize
        model=self.bank.models['detector'];hooks=[]
        def forbidden(*args):raise AssertionError('Box/presence head executed in classification-only mode')
        for module in (model.box_reduce,model.box_fc,model.presence):
            hooks.append(module.register_forward_hook(forbidden))
        try:
            image=Image.fromarray(cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB))
            with torch.inference_mode():
                expected=model.classify(tensorize(letterbox(image)[0]).unsqueeze(0)).softmax(1)[0]
            result=self.bank.predict(self.frame,'detector')
            np.testing.assert_allclose(result['all_class_scores'],expected.numpy(),atol=1e-7)
        finally:
            for hook in hooks:hook.remove()

    def test_missing_detector_does_not_disable_original(self):
        bank=ModelBank(torch.device('cpu'),detector=PROJECT/'recognition'/'nonexistent.pth')
        self.assertIn('original',bank.models);self.assertNotIn('detector',bank.models)
        self.assertIn('detector',bank.unavailable)
        self.assertIn('class_score',bank.predict(self.frame,'original'))

    def test_display_does_not_draw_rectangles(self):
        frame=cv2.resize(self.frame,(640,480))
        result=self.bank.predict(self.frame,'original')
        with patch('cv2.rectangle',side_effect=AssertionError('No bounding rectangles allowed')):
            render(frame,self.bank,'original',False,result)

    def test_switch_discards_old_inflight_prediction(self):
        entered=threading.Event();release=threading.Event()
        class FakeBank:
            models={'original':None,'detector':None}
            def predict(self,frame,key,shared):
                if key=='original':entered.set();release.wait(2)
                return {'model':key}
        worker=PredictionWorker(FakeBank())
        try:
            worker.submit(self.frame,1,time.perf_counter())
            self.assertTrue(entered.wait(1))
            worker.select('detector');release.set()
            worker.submit(self.frame,2,time.perf_counter())
            deadline=time.perf_counter()+2
            while worker.snapshot()[2] is None and time.perf_counter()<deadline:time.sleep(.01)
            result=worker.snapshot()[2]
            self.assertIsNotNone(result);self.assertEqual(result['model'],'detector')
            self.assertEqual(result['frame_id'],2)
        finally:
            release.set();worker.close()


if __name__=='__main__':unittest.main()
