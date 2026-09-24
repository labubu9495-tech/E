"""Focused checks for missing/padding handling and causal token masking."""
import unittest
import numpy as np
import torch
from data import adapt,interval_mask,condition_observed,TextEncoder
from models import Predictor,gap_features


class Invariants(unittest.TestCase):
    def test_masked_token_cannot_leak_through_bert(self):
        torch.set_num_threads(1)
        tokens=np.zeros((1,3,50),dtype=np.int64)
        tokens[0,0,:4]=[101,2000,2001,102];tokens[0,1,:4]=1
        seen=np.zeros((1,50),bool);seen[0,1]=True
        encoder=TextEncoder('cpu')
        a=encoder.encode(tokens,seen)
        tokens[0,0,2]=2307
        b=encoder.encode(tokens,seen)
        self.assertTrue(np.array_equal(a,b))
        self.assertTrue(np.all(a[~seen]==0))

    def test_internal_hole_does_not_shorten_sequence(self):
        tokens=np.zeros((1,3,50));tokens[0,0,:7]=[101,1030,0,0,2020,3000,102]
        tokens[0,1,:7]=[1,1,0,0,1,1,1]
        a=adapt(dict(text_bert=tokens,audio=np.zeros((1,50,74)),vision=np.zeros((1,50,35))))
        self.assertEqual(a['sequence'].sum(),5)
        self.assertEqual(a['observed'][:,:,0].sum(),3)
        self.assertTrue(a['sequence'][0,2])

    def test_gap_and_padding(self):
        p=torch.tensor([[False,True,True,True,True,False]])
        o=torch.zeros((1,6,3),dtype=torch.bool);o[:,1]=True;o[:,4]=True
        gap=gap_features(o,p)
        self.assertTrue(torch.allclose(gap[0,:,0],torch.log1p(torch.tensor([0.,0.,1.,2.,0.,0.]))))

    def test_hidden_content_padding_and_empty(self):
        torch.set_num_threads(1)
        torch.manual_seed(2)
        xs=[torch.randn(2,8,d) for d in [256,74,35]]
        p=torch.tensor([[False,True,True,True,True,False,False,False]]*2)
        o=p.unsqueeze(-1).expand(-1,-1,3).clone();o[:,2:4,1]=False
        for kind in ['gru','state','no_history','smooth','mlp','text']:
            m=Predictor(kind=kind,dropout=0).eval()
            expected=m(*xs,p,o)
            changed=[x.clone() for x in xs]
            for j,x in enumerate(changed):
                x[~o[:,:,j]]=12345
            actual=m(*changed,p,o)
            for a,b in zip(expected,actual):
                self.assertTrue(torch.allclose(a,b,atol=1e-6),kind)
            longer=[torch.cat([x,torch.randn(2,3,x.shape[-1])],1) for x in xs]
            actual=m(*longer,torch.cat([p,torch.zeros(2,3,dtype=torch.bool)],1),torch.cat([o,torch.zeros(2,3,3,dtype=torch.bool)],1))
            for a,b in zip(expected,actual):
                self.assertTrue(torch.allclose(a,b,atol=1e-6),kind)
            empty=m(*xs,p,torch.zeros_like(o))
            self.assertTrue(torch.isfinite(empty[0]).all())
            self.assertTrue(torch.allclose(empty[0].softmax(-1),m.class_prior.expand(2,-1)))

    def test_matched_block_budget(self):
        p=np.zeros((3,50),bool);p[:,1:22]=True
        obs=np.repeat(p[:,:,None],3,axis=2)
        base=dict(sequence=p,observed=obs)
        counts=[]
        for loc in ['middle','two_blocks']:
            result=condition_observed(base,dict(modalities=[1],ratio=.3,location=loc))
            counts.append((obs & ~result).sum())
        self.assertEqual(*counts)


if __name__=='__main__':
    unittest.main()
