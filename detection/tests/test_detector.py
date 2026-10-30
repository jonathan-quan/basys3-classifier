import copy
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from classifier.model import TinyClassifier
from classifier.common import seed_everything
from detection import PROJECT
from detection.model import TinyDetector,decode,profile,corners,with_head_batch_norm,with_backbone_batch_norm,export_inference
from detection.data import letterbox,inverse_box,DetectionDataset,validate_rows,tensorize,augment
from detection.metrics import aligned_iou,compute
from detection.train import objective
from detection.fetch_data import fingerprints,save_jsonl


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):seed_everything(42)

    def test_exact_profile_and_transfer_parity(self):
        p=profile();self.assertEqual((p['parameters'],p['weights'],p['biases'],p['macs']),(7681,7408,273,886704))
        m=TinyDetector();report=m.transfer_classifier(PROJECT/'best_model.pth')
        self.assertEqual((report['parameters_transferred'],report['tensors_transferred']),(6084,20))
        c=torch.load(PROJECT/'best_model.pth',map_location='cpu',weights_only=True)
        old=TinyClassifier('context');old.load_state_dict(c['model_state_dict']);old.eval();m.eval()
        x=torch.randn(3,3,96,96)
        with torch.inference_mode():self.assertTrue(torch.equal(m(x)['class_logits'],old(x)))

    def test_geometry_roundtrip_and_pixel_contract(self):
        for w,h in [(640,480),(127,641),(641,127),(101,101)]:
            box=np.array([.11,.2,.85,.91]);im=Image.new('RGB',(w,h),(128,0,255))
            out,b,t=letterbox(im,box);got=inverse_box(b,t)/[w,h,w,h]
            np.testing.assert_allclose(box,got,atol=1e-6)
            x=tensorize(out);self.assertEqual(tuple(x.shape),(3,96,96))
            self.assertTrue(torch.isfinite(x).all());self.assertGreaterEqual(x.min(),-1)
            self.assertLessEqual(x.max(),127/128)

    def test_head_normalization_folds_without_changing_outputs(self):
        model=with_head_batch_norm(TinyDetector());model.train()
        for _ in range(3):model(torch.randn(4,3,96,96))
        model.eval();x=torch.randn(2,3,96,96)
        with torch.inference_mode():before=model(x);folded=export_inference(model);after=folded(x)
        self.assertEqual(sum(p.numel() for p in model.parameters()),7689)
        self.assertEqual(sum(p.numel() for p in folded.parameters()),7681)
        for k in before:torch.testing.assert_close(before[k],after[k],rtol=1e-4,atol=1e-5)
        self.assertFalse(any(isinstance(m,torch.nn.BatchNorm2d) for m in folded.modules()))
        model.train();mean=model.box_reduce[1].running_mean.clone()
        model.classify(x)
        self.assertTrue(torch.equal(mean,model.box_reduce[1].running_mean))

    def test_backbone_normalization_identity_transfer_and_folding(self):
        model=TinyDetector();model.transfer_classifier(PROJECT/'best_model.pth');model.eval()
        x=torch.randn(3,3,96,96)
        with torch.inference_mode():before=model(x)
        model=with_backbone_batch_norm(model).eval()
        with torch.inference_mode():after=model(x)
        for k in before:torch.testing.assert_close(before[k],after[k],rtol=1e-4,atol=1e-5)
        model.train()
        for _ in range(3):model(torch.randn(4,3,96,96))
        model.eval();folded=export_inference(model)
        with torch.inference_mode():a=model(x);b=folded(x)
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=1e-4,atol=1e-5)
        restored=TinyDetector();restored.load_state_dict(folded.state_dict(),strict=True)
        self.assertEqual(sum(p.numel() for p in folded.parameters()),7681)

    def test_augmentation_boxes_remain_valid(self):
        im=Image.new('RGB',(96,96),'white');box=np.array([.1,.2,.8,.9],dtype=np.float32)
        for _ in range(50):
            image,b=augment(im,box)
            self.assertEqual(image.size,(96,96));self.assertTrue(np.all(b>=0) and np.all(b<=1));self.assertTrue(np.all(b[2:]>b[:2]))

    def test_no_candidate_training_or_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.png';Image.new('RGB',(96,96)).save(path)
            r={'id':'a','path':str(path),'label':0,'present':1,'bbox':[.1,.1,.9,.9],
               'annotation_status':'candidate','group':'g','pixel_sha256':'p','split':'train',
               'sha256':fingerprints(path)['sha256'],'source':'Synthetic unit fixture','license':'CC0'}
            with self.assertRaisesRegex(ValueError,'candidate'):DetectionDataset([r],'train')
            r['annotation_status']='source_ground_truth';rows=[]
            for split in ('train','val','test'):
                for label in range(4):
                    q={**r,'id':f'{split}{label}','group':f'g{split}{label}','pixel_sha256':f'p{split}{label}','split':split,'label':label};rows.append(q)
                rows.append({**r,'id':f'{split}n','label':-1,'present':0,'bbox':None,'group':f'gn{split}','pixel_sha256':f'pn{split}','split':split})
            validate_rows(rows)
            bad=copy.deepcopy(rows);bad[-1]['group']=bad[0]['group']
            with self.assertRaisesRegex(ValueError,'leakage'):validate_rows(bad)
            bad=copy.deepcopy(rows);bad[-1]['pixel_sha256']=bad[0]['pixel_sha256']
            with self.assertRaisesRegex(ValueError,'leakage'):validate_rows(bad)

    def test_metrics_count_wrong_class_and_bad_box(self):
        base={'present':1,'label':0,'predicted_class':0,'score':.9,'presence_score':.9,'iou':.8}
        records=[base,{**base,'predicted_class':1},{**base,'iou':.1},
                 {**base,'present':0,'label':-1,'iou':None}]
        m=compute(records,.5,.5)
        self.assertEqual((m['detection_tp'],m['detection_fp'],m['detection_fn']),(1,3,2))
        self.assertAlmostEqual(m['presence_precision'],.75)
        self.assertAlmostEqual(m['mean_iou_all_positives'],1.7/3)
        self.assertIsNone(m['per_class']['book']['ap50'])

    def test_heads_receive_useful_gradients_and_learn(self):
        m=TinyDetector();m.transfer_classifier(PROJECT/'best_model.pth')
        for p in m.features.parameters():p.requires_grad_(False)
        for p in m.fc.parameters():p.requires_grad_(False)
        x=torch.randn(4,3,96,96)*.1
        target={'image':x,'present':torch.tensor([1.,1.,0.,0.]),'label':torch.tensor([0,1,-1,-1]),
                'box':torch.tensor([[.1,.2,.7,.8],[.2,.1,.9,.7],[0,0,0,0],[0,0,0,0]])}
        opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.01)
        initial=float(objective(m(x),target,torch.device('cpu'))[0].detach())
        for _ in range(30):
            opt.zero_grad();loss,_=objective(m(x),target,torch.device('cpu'));loss.backward();opt.step()
        final=float(objective(m(x),target,torch.device('cpu'))[0].detach())
        self.assertLess(final,initial)
        out=decode(m(x));self.assertTrue(torch.all(out['boxes']>=0));self.assertTrue(torch.all(out['boxes']<=1))
        self.assertTrue(m.box_fc.weight.grad.abs().sum()>0)
        self.assertTrue(m.presence.weight.grad.abs().sum()>0)

    def test_synthetic_train_export_evaluate_workflow(self):
        from detection.train import train
        from detection.evaluate import evaluate
        with tempfile.TemporaryDirectory(prefix='detector_smoke_') as folder:
            folder=Path(folder);rows=[]
            for split in ('train','val','test'):
                for label in (-1,0,1,2,3):
                    uid=f'{split}_{label}';path=folder/(uid+'.png')
                    im=Image.new('RGB',(96,96),(10+len(rows),20,30))
                    if label>=0:
                        from PIL import ImageDraw
                        ImageDraw.Draw(im).rectangle((20,25,70,80),fill=(40+label*40,80,120))
                    im.save(path)
                    rows.append({'id':uid,'path':str(path),'label':label,'present':int(label>=0),
                        'bbox':[20/96,25/96,70/96,80/96] if label>=0 else None,'split':split,
                        'group':uid,'source':'Explicit synthetic smoke fixture','license':'CC0',
                        'annotation_status':'source_ground_truth','synthetic':True,**fingerprints(path)})
            manifest=folder/'manifest.jsonl';save_jsonl(manifest,rows);output=folder/'run'
            args=Namespace(output=str(output),epochs=1,warmup=0,patience=1,manifest=str(manifest),
                pilot=False,smoke_test=True,seed=42,device='cpu',classifier=str(PROJECT/'best_model.pth'),
                batch_size=8,no_classifier_rehearsal=True,head_lr=.001,backbone_lr=.0001,head_batch_norm=True,backbone_batch_norm=True,schedule='cosine')
            train(args)
            result=evaluate(output/'best_detector.pth',manifest,'test',folder/'evaluation',torch.device('cpu'))
            self.assertEqual(result['images'],5)
            checkpoint=torch.load(output/'best_detector.pth',map_location='cpu',weights_only=True)
            self.assertTrue(checkpoint['smoke_test'])
            self.assertTrue(checkpoint['training_head_batch_norm_folded'])


if __name__=='__main__':unittest.main()
