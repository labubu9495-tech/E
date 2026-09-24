import unittest
import numpy as np
from features import visible_documents,visible_ngrams,pooled_features


class FeatureChecks(unittest.TestCase):
    def test_missing_does_not_form_false_bigram(self):
        tokens=np.array([[101,2000,100,2001,102,0]])
        sequence=np.array([[False,True,True,True,False,False]])
        observed=np.array([[False,True,False,True,False,False]])
        doc=visible_documents(tokens,observed,sequence)[0]
        self.assertEqual(list(visible_ngrams(doc)),['u2000','u2001'])
        self.assertEqual(list(visible_ngrams(['2000','2001'])),['u2000','b2000_2001','u2001'])

    def test_pooling_ignores_hidden_content_and_padding(self):
        rng=np.random.default_rng(2)
        xs=[rng.normal(size=(2,6,d)).astype(np.float32) for d in [256,74,35]]
        seq=np.array([[False,True,True,True,False,False]]*2)
        obs=np.repeat(seq[:,:,None],3,axis=2);obs[:,2,0]=False;obs[1]=False
        result=pooled_features(*xs,seq,obs)
        changed=[x.copy() for x in xs]
        for i,x in enumerate(changed):x[~obs[:,:,i]]=np.nan
        np.testing.assert_array_equal(result,pooled_features(*changed,seq,obs))
        self.assertTrue(np.isfinite(result).all());self.assertTrue((result[1]==0).all())
        longer=[np.pad(x,((0,0),(0,2),(0,0)),constant_values=99) for x in xs]
        np.testing.assert_array_equal(result,pooled_features(*longer,np.pad(seq,((0,0),(0,2))),np.pad(obs,((0,0),(0,2),(0,0)))))


if __name__=='__main__':unittest.main()
