import unittest
import tempfile
from pathlib import Path
import torch
from paper_models import make_model, restoration_losses


class PaperModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def example(self, n=3, length=12):
        torch.manual_seed(7)
        xs = [torch.randn(n, length, d) for d in (256, 74, 35)]
        sequence = torch.zeros(n, length, dtype=torch.bool)
        sequence[:, 1:9] = True
        observed = sequence[..., None].expand(-1, -1, 3).clone()
        observed[:, 3:6, 0] = False
        observed[0, :, 2] = False
        return xs + [sequence, observed]

    def test_hidden_content_and_padding_invariance(self):
        for name in ('mult', 'emt_dlfr'):
            with self.subTest(name=name):
                model = make_model(name).eval()
                inputs = self.example()
                with torch.no_grad():
                    expected = model(*inputs)
                    changed = [x.clone() for x in inputs]
                    for i in range(3):
                        changed[i][~inputs[4][..., i]] = 9999.
                    actual = model(*changed)
                    for a, b in zip(expected, actual):
                        torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-6)
                    appended = [torch.cat([x, torch.randn(len(x), 4, x.shape[-1])], 1) for x in inputs[:3]]
                    appended += [torch.cat([inputs[3], torch.zeros(3, 4, dtype=torch.bool)], 1),
                                 torch.cat([inputs[4], torch.zeros(3, 4, 3, dtype=torch.bool)], 1)]
                    for a, b in zip(expected, model(*appended)):
                        torch.testing.assert_close(a, b, rtol=2e-5, atol=3e-6)

    def test_all_missing_and_single_sample_checkpoint(self):
        for name in ('mult', 'emt_dlfr'):
            model = make_model(name).eval()
            inputs = self.example(n=1)
            inputs[4].zero_()
            with torch.no_grad():
                logits, reg = model(*inputs)
                torch.testing.assert_close(logits.softmax(-1)[0], model.class_prior)
                torch.testing.assert_close(reg[0], model.target_median)
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / 'model.pt'
                torch.save(model.state_dict(), p)
                restored = make_model(name).eval()
                restored.load_state_dict(torch.load(p, weights_only=True))
                with torch.no_grad():
                    for a, b in zip(model(*inputs), restored(*inputs)):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_backward_with_missing_modality_is_finite(self):
        for name in ('mult', 'emt_dlfr'):
            model = make_model(name).train()
            inputs = self.example()
            inputs[4][1] = False
            logits, reg = model(*inputs)
            loss = logits.square().mean() + reg.square().mean()
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients), name)
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)

    def test_reconstruction_uses_only_artificially_hidden_observations(self):
        model = make_model('emt_dlfr').eval()
        complete = self.example()
        missing = [x.clone() for x in complete]
        missing[4][:, 6:8, 1] = False
        ma = model(*missing, return_aux=True)[2]
        ca = model(*complete, return_aux=True)[2]
        recon, attra = restoration_losses(ma, ca, complete, missing[4])
        self.assertTrue(torch.isfinite(recon + attra))
        altered = [x.clone() for x in complete]
        hidden = complete[4] & ~missing[4]
        for i in range(3):
            altered[i][~hidden[..., i]] = 12345.
        recon2, _ = restoration_losses(ma, ca, altered, missing[4])
        torch.testing.assert_close(recon, recon2, rtol=0, atol=0)
        recon0, _ = restoration_losses(ma, ca, complete, complete[4])
        self.assertEqual(float(recon0.detach()), 0.)
        for i in range(3):
            complete[i].requires_grad_(True)
        recon, _ = restoration_losses(ma, ca, complete, missing[4])
        recon.backward()
        self.assertTrue(all(x.grad is None for x in complete[:3]))

    def test_emt_restore_and_ablation_have_identical_prediction_architecture(self):
        torch.manual_seed(17)
        a = make_model('emt_dlfr').eval()
        torch.manual_seed(17)
        b = make_model('emt_no_restore').eval()
        inputs = self.example()
        with torch.no_grad():
            for x, y in zip(a(*inputs), b(*inputs)):
                torch.testing.assert_close(x, y, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
