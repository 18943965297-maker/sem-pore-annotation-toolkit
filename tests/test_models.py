import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional torch dependency is not installed")
class ModelTests(unittest.TestCase):
    def test_architecture_gradient_and_preservation(self):
        from models.aedge import AedgeNet
        from models.fusion import EdgeSAMCorrection
        from models.fusion_v2 import EdgeSAMV2Supervised, supervised_loss

        torch.set_num_threads(1)
        torch.manual_seed(7)
        image = torch.rand(2, 3, 32, 32)
        ae = AedgeNet(base_channels=8)
        out = ae(image)
        self.assertEqual(out["mask"].shape, (2, 1, 32, 32))
        out["mask"].square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in ae.parameters()))
        ap = torch.randn(2, 1, 32, 32)
        ae_logits, edge = out["mask"].detach(), out["edge"].detach()
        original = EdgeSAMCorrection()(image, ap, ae_logits, edge)
        torch.testing.assert_close(original["logits"], ap)
        fusion = EdgeSAMV2Supervised()
        corrected = fusion(image, ap, ae_logits, edge)
        self.assertTrue(torch.all(corrected["probability"][ap >= 0] == 1))
        outside = ~corrected["candidate"] & ~(ap >= 0)
        self.assertTrue(torch.all(corrected["probability"][outside] == 0))
        loss = supervised_loss(corrected, (torch.rand_like(ap) > .5).float())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in fusion.parameters()))
