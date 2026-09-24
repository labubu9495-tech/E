"""Focused checks for missing/padding handling and causal token masking."""
import unittest
import numpy as np
import torch
from data import adapt,interval_mask,condition_observed,TextEncoder,normalize,observation_audit
from models import Predictor,gap_features


class Invariants(unittest.TestCase):
    @staticmethod
    def example():
        tokens=np.zeros((1,3,50),dtype=np.float32)
        tokens[0,0,:6]=[101,2000,100,2001,2002,102];tokens[0,1,:6]=1
        audio=np.zeros((1,50,74),np.float32);audio[:,1:5]=1;audio[:,2]=0
        vision=np.zeros((1,50,35),np.float32);vision[:,1:5]=1;vision[:,2]=0
        return dict(text_bert=tokens,audio=audio,vision=vision)

    def test_official_unk_is_missing_but_preserves_position(self):
        source=self.example();a=adapt(source)
        self.assertTrue(a['sequence'][0,2])
        self.assertFalse(a['observed'][0,2].any())
        self.assertEqual(a['sequence'].sum(),4)
        self.assertTrue(a['observed'][0,3:].any())
        self.assertTrue(adapt(source,text_missing_policy='ordinary_unk')['observed'][0,2,0])
        self.assertEqual(source['text_bert'][0,0,2],100)  # Original data untouched.
        info=observation_audit(a,0)
        self.assertEqual(info['text_unavailable_fraction'],.25)
        self.assertEqual(info['audio_unavailable_intervals'],[[2,3]])

    def test_unk_removed_before_contextual_encoding(self):
        torch.set_num_threads(1);a=adapt(self.example());encoder=TextEncoder('cpu')
        actual=encoder.encode(a['tokens'],a['observed'][:,:,0])
        erased=a['tokens'].copy();erased[:,:,2]=0
        expected=encoder.encode(erased,a['observed'][:,:,0])
        self.assertTrue(np.array_equal(actual,expected))
        self.assertTrue(np.all(actual[:,2]==0))

    def test_mask_precedes_normalization_and_zero_vision_is_safe(self):
        source=self.example();source['vision'][:]=0;a=adapt(source)
        mean=np.ones(74,dtype=np.float32)*4;std=np.ones(74,dtype=np.float32)*2
        z=normalize(a['audio'],mean,std,a['observed'][:,:,1])
        self.assertTrue(np.all(z[~a['observed'][:,:,1]]==0))
        self.assertTrue(np.all(z[a['observed'][:,:,1]]==-1.5))
        self.assertIn('vision_entirely_unavailable',observation_audit(a,0)['quality_flags'])

    def test_float_tokens_require_exact_integer_values(self):
        source=self.example();source['text_bert'][0,0,1]=2000.5
        with self.assertRaises(ValueError):adapt(source)

    def test_zero_missing_ratio_and_natural_order(self):
        from pathlib import Path
        from infer import natural_file_key
        self.assertFalse(interval_mask(np.ones((2,50),bool),0).any())
        names=[Path(x) for x in ['10.pkl','2.pkl','1.pkl']]
        self.assertEqual([p.name for p in sorted(names,key=natural_file_key)],['1.pkl','2.pkl','10.pkl'])

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
