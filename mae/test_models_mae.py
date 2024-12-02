import unittest
import torch
from mae.models_mae import MaskedAutoencoderViT

import torch.nn.functional as F

class TestObjectMasking(unittest.TestCase):
    def setUp(self):
        self.model = MaskedAutoencoderViT()
        self.N = 2  # batch size
        self.L = 196  # sequence length (e.g., 14x14 patches)
        self.D = 768  # embedding dimension
        self.H = self.W = 224  # image dimensions
        self.mask_ratio = 0.75

        # Create dummy inputs
        self.x = torch.randn(self.N, self.L, self.D)
        self.imgs_mask = torch.randint(0, 10, (self.N, self.H, self.W))
        self.co_occur = torch.randint(0, 10, (self.N, 2))

    def test_object_masking_shape(self):
        x_masked, mask, ids_restore = self.model.object_masking(self.x, self.imgs_mask, self.co_occur, self.mask_ratio)
        len_keep = int(self.L * (1 - self.mask_ratio))

        self.assertEqual(x_masked.shape, (self.N, len_keep, self.D))
        self.assertEqual(mask.shape, (self.N, self.L))
        self.assertEqual(ids_restore.shape, (self.N, self.L))

    def test_object_masking_values(self):
        x_masked, mask, ids_restore = self.model.object_masking(self.x, self.imgs_mask, self.co_occur, self.mask_ratio)
        len_keep = int(self.L * (1 - self.mask_ratio))

        # Check that mask contains correct number of masked elements
        for i in range(self.N):
            self.assertEqual(mask[i].sum().item(), self.L - len_keep)

        # Check that ids_restore is a permutation of range(L)
        for i in range(self.N):
            self.assertTrue(torch.equal(ids_restore[i].sort().values, torch.arange(self.L, device=ids_restore.device)))

    def test_object_masking_no_co_occur(self):
        # Test case where there are no co-occurring objects
        co_occur = torch.zeros_like(self.co_occur)
        x_masked, mask, ids_restore = self.model.object_masking(self.x, self.imgs_mask, co_occur, self.mask_ratio)
        len_keep = int(self.L * (1 - self.mask_ratio))

        # Check that mask contains correct number of masked elements
        for i in range(self.N):
            self.assertEqual(mask[i].sum().item(), self.L - len_keep)

if __name__ == '__main__':
    unittest.main()