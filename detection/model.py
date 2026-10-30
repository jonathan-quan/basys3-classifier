"""Exact 7,681-parameter single-object detector; no pretrained large model."""
import copy
from collections import OrderedDict
import torch
from torch import nn

from classifier.model import TinyClassifier
from classifier.common import file_hash
from . import CLASSES, SIZE


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        classifier=TinyClassifier("context")
        self.features=classifier.features
        self.pool=classifier.pool
        self.fc=classifier.fc
        self.presence=nn.Linear(48,1)
        self.box_reduce=nn.Conv2d(48,8,1)
        self.box_relu=nn.ReLU()
        self.box_fc=nn.Linear(8*6*6,4)
        nn.init.kaiming_normal_(self.box_reduce.weight,nonlinearity="relu")
        nn.init.zeros_(self.box_reduce.bias)
        nn.init.normal_(self.box_fc.weight,std=0.002)
        nn.init.zeros_(self.box_fc.bias)
        nn.init.zeros_(self.presence.weight)
        nn.init.zeros_(self.presence.bias)
        assert sum(p.numel() for p in self.parameters())==7681

    def forward(self,images):
        if images.shape[-2:]!=(SIZE,SIZE):
            raise ValueError("Detector requires exactly 96x96 RGB")
        features=self.features(images)
        pooled=self.pool(features).flatten(1)
        return {"class_logits":self.fc(pooled),
                "presence_logits":self.presence(pooled).squeeze(1),
                "box_cxcywh":self.box_fc(self.box_relu(self.box_reduce(features)).flatten(1)).sigmoid()}

    def transfer_classifier(self,path):
        checkpoint=torch.load(path,map_location="cpu",weights_only=True)
        if checkpoint.get("architecture")!="context" or checkpoint.get("classes")!=list(CLASSES) or checkpoint.get("input_size")!=96 or checkpoint.get("smoke_test"):
            raise ValueError("Expected the real 6,084-parameter context classifier")
        state=checkpoint["model_state_dict"]
        result=self.load_state_dict(state,strict=False)
        expected={"presence.weight","presence.bias","box_reduce.weight","box_reduce.bias","box_fc.weight","box_fc.bias"}
        if set(result.missing_keys)!=expected or result.unexpected_keys:
            raise ValueError(f"Unexpected transfer mismatch: {result}")
        current=self.state_dict()
        if not all(torch.equal(current[k],v) for k,v in state.items()):
            raise AssertionError("Transferred tensors changed")
        count=sum(v.numel() for v in state.values())
        assert count==6084
        return {"checkpoint_sha256":file_hash(path),"parameters_transferred":count,
                "tensors_transferred":len(state),"new_parameters":1597}

    def classify(self,images):
        # Rehearsal has class labels only. Do not update localization-head BN
        # statistics with this different, unboxed image distribution.
        return self.fc(self.pool(self.features(images)).flatten(1))


def corners(boxes):
    center=boxes[...,:2];half=boxes[...,2:]/2
    return torch.cat((center-half,center+half),dim=-1).clamp(0,1)


def decode(outputs):
    class_prob=outputs["class_logits"].softmax(-1)
    value,label=class_prob.max(-1)
    presence=outputs["presence_logits"].sigmoid()
    return {"boxes":corners(outputs["box_cxcywh"]),"classes":label,
            "presence":presence,"class_scores":value,"scores":presence*value}


def load_detector(path,device):
    c=torch.load(path,map_location="cpu",weights_only=True)
    if c.get("architecture")!="reuse_7681" or c.get("classes")!=list(CLASSES) or c.get("input_size")!=SIZE:
        raise ValueError("Not a compatible detector checkpoint")
    m=TinyDetector();m.load_state_dict(c["model_state_dict"],strict=True)
    return m.to(device).eval(),c


def with_head_batch_norm(model):
    """Training-only normalization; fold into the new PW head before export."""
    conv=model.box_reduce
    if not isinstance(conv,nn.Conv2d):raise ValueError('Head already wrapped')
    conv.register_parameter('bias',None)
    model.box_reduce=nn.Sequential(conv,nn.BatchNorm2d(8))
    return model


def with_backbone_batch_norm(model):
    """Identity-initialized training BN; keep every transferred conv bias/tensor."""
    layers=OrderedDict()
    for name,module in model.features._modules.items():
        layers[name]=module
        if isinstance(module,nn.Conv2d):
            bn=nn.BatchNorm2d(module.out_channels)
            # In eval, BN starts as the identity including epsilon correction.
            nn.init.constant_(bn.weight,(1+bn.eps)**.5)
            layers['train_bn_'+name]=bn
    model.features=nn.Sequential(layers)
    return model


def export_inference(model):
    result=copy.deepcopy(model).cpu().eval()
    layers=list(result.features._modules.items());folded=OrderedDict();i=0
    while i<len(layers):
        name,module=layers[i]
        if isinstance(module,nn.Conv2d) and i+1<len(layers) and isinstance(layers[i+1][1],nn.BatchNorm2d):
            module=torch.nn.utils.fusion.fuse_conv_bn_eval(module,layers[i+1][1]);i+=1
        folded[name]=module;i+=1
    result.features=nn.Sequential(folded)
    if isinstance(result.box_reduce,nn.Sequential):
        conv,bn=result.box_reduce
        result.box_reduce=torch.nn.utils.fusion.fuse_conv_bn_eval(conv,bn)
    assert sum(p.numel() for p in result.parameters())==7681
    return result


def profile():
    model=TinyDetector();rows=[];hooks=[]
    def record(name):
        def hook(m,inp,out):
            k=m.in_features if isinstance(m,nn.Linear) else m.in_channels//m.groups*m.kernel_size[0]*m.kernel_size[1]
            rows.append({"name":name,"shape":list(out.shape[1:]),"macs":out.numel()*k})
        return hook
    for name,m in model.named_modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)):hooks.append(m.register_forward_hook(record(name)))
    with torch.inference_mode():model(torch.zeros(1,3,96,96))
    for h in hooks:h.remove()
    return {"parameters":sum(p.numel() for p in model.parameters()),"weights":sum(p.numel() for n,p in model.named_parameters() if n.endswith('weight')),
            "biases":sum(p.numel() for n,p in model.named_parameters() if n.endswith('bias')),
            "macs":sum(r['macs'] for r in rows),"layers":rows,"int8_measured":False}


if __name__=="__main__":
    import json
    print(json.dumps(profile(),indent=2))
