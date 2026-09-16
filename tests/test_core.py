import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestData(unittest.TestCase):
    def test_training_views_are_sam_only(self):
        from localization_lens.training import VIEW_TYPES

        self.assertEqual(VIEW_TYPES, {
            "unicolor": ("sam_unicolor.png",),
            "multicolor": ("sam_multicolor.png",),
            "masked": ("sam_masked.png",),
        })

    def test_infer_organ(self):
        from localization_lens.data import infer_organ

        self.assertEqual(infer_organ("Is the left lung enlarged?"), "lung")
        self.assertEqual(infer_organ("What is shown?"), "radiology anatomy")

    def test_grouped_split_has_no_leakage(self):
        from localization_lens.data import image_grouped_resplit

        rows = [{"image_id": value} for value in ["a", "a", "b", "c", "c"]]
        image_grouped_resplit(rows, 0.30, 3)
        by_id = {}
        for row in rows:
            by_id.setdefault(row["image_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in by_id.values()))

    def test_stratified_limit_keeps_train_and_test(self):
        from localization_lens.io import stratified_image_limit

        rows = [
            {"image_id": f"train-{i}", "split": "train"} for i in range(8)
        ] + [{"image_id": f"test-{i}", "split": "test"} for i in range(2)]
        selected = stratified_image_limit(rows, 5)
        self.assertEqual(len(selected), 5)
        self.assertEqual({row["split"] for row in selected}, {"train", "test"})

    def test_token_recall(self):
        from localization_lens.inference import token_recall

        self.assertEqual(token_recall("left lung", "left lower lung"), 2 / 3)


class TestViews(unittest.TestCase):
    def test_background_and_peripheral_label_filter(self):
        import numpy as np
        from PIL import Image
        from localization_lens.views import filter_background_annotations

        pixels = np.zeros((100, 100), dtype=np.uint8)
        pixels[20:80, 20:80] = 90
        background = np.zeros_like(pixels, dtype=bool)
        background[30:60, :10] = True
        label = np.zeros_like(background)
        label[2:8, 80:95] = True
        pixels[label] = 240
        anatomy = np.zeros_like(background)
        anatomy[30:60, 30:60] = True
        # A dark interior lung-like region must not be rejected merely for
        # moderate darkness, or a bright device mistaken for border text.
        pixels[anatomy] = 35
        device = np.zeros_like(background)
        device[40:50, 65:75] = True
        pixels[device] = 240
        kept = filter_background_annotations(Image.fromarray(pixels),
                                             [background, label, anatomy, device])
        self.assertEqual(len(kept), 2)
        np.testing.assert_array_equal(kept[0], anatomy)
        np.testing.assert_array_equal(kept[1], device)

    def test_mask_extent_limit_checks_both_dimensions(self):
        import numpy as np
        from localization_lens.views import normalize_masks

        valid = np.zeros((100, 200), dtype=bool)
        valid[10:70, 20:140] = True  # Exactly 60% in both dimensions.
        tall = valid.copy()
        tall[70, 20:140] = True
        wide = valid.copy()
        wide[10:70, 140] = True
        sparse_wide = np.zeros_like(valid)
        sparse_wide[40:42, 10:150] = True
        selected = normalize_masks([valid, tall, wide, sparse_wide], valid.shape)
        self.assertEqual(len(selected), 1)
        np.testing.assert_array_equal(selected[0], valid)

    def test_internal_crop_edges_rejected_but_image_edges_allowed(self):
        import numpy as np
        from localization_lens.sam_augment import touches_internal_crop_edge

        mask = np.ones((100, 100), dtype=bool)
        self.assertFalse(touches_internal_crop_edge(mask, (0, 0, 100, 100), (100, 100)))
        self.assertTrue(touches_internal_crop_edge(mask, (0, 0, 100, 100), (200, 200)))
        mask[:] = False
        mask[30:60, 30:60] = True
        self.assertFalse(touches_internal_crop_edge(mask, (50, 50, 150, 150), (200, 200)))

    def test_overlapping_masks_do_not_saturate_or_darken_background(self):
        import numpy as np
        from PIL import Image
        from localization_lens.views import render_views, PALETTE

        image = Image.new("RGB", (32, 32), (100, 100, 100))
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        with TemporaryDirectory() as directory:
            outputs = render_views(image, [mask] * 8, Path(directory), "sam", 0.5)
            uni = np.asarray(Image.open(outputs["sam_unicolor"]))
            multi = np.asarray(Image.open(outputs["sam_multicolor"]))
            np.testing.assert_array_equal(uni[10, 10], ((100 + PALETTE[0]) / 2).astype(np.uint8))
            np.testing.assert_array_equal(multi[10, 10], ((100 + PALETTE[7]) / 2).astype(np.uint8))
            np.testing.assert_array_equal(multi[0, 0], [100, 100, 100])

    def test_sam_crop_passes_are_sequential_and_restore_full_size(self):
        import numpy as np
        from argparse import Namespace
        from PIL import Image
        from localization_lens.sam_augment import generate_masks

        sizes = []
        def fake_generator(image, **kwargs):
            sizes.append(image.size)
            self.assertEqual(kwargs["crops_n_layers"], 0)
            mask = np.zeros((image.height, image.width), dtype=bool)
            mask[: max(1, image.height // 2), : max(1, image.width // 2)] = True
            return {"masks": [mask], "scores": [0.9]}

        args = Namespace(crops_n_layers=1, points_per_batch=16, points_per_crop=64,
                         pred_iou_thresh=0.88, stability_score_thresh=0.95)
        masks = generate_masks(fake_generator, Image.new("RGB", (1024, 1291)), args)
        self.assertEqual(len(sizes), 5)
        self.assertGreater(len(set(sizes)), 1)
        self.assertTrue(all(mask.shape == (1291, 1024) for mask in masks))

    def test_render_views_writes_complete_lens(self):
        try:
            import numpy as np
            from PIL import Image
            from localization_lens.views import render_views
        except ImportError:
            self.skipTest("numpy and Pillow are required")
        image = Image.new("RGB", (16, 16), "white")
        first = np.zeros((16, 16), dtype=bool)
        second = np.zeros((16, 16), dtype=bool)
        first[2:8, 2:8] = True
        second[8:14, 8:14] = True
        with TemporaryDirectory() as directory:
            outputs = render_views(image, [first, second], Path(directory), "sam", 0.5)
            self.assertEqual(len(outputs), 5)
            self.assertTrue(all(Path(path).is_file() for path in outputs.values()))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class TestModeling(unittest.TestCase):
    def test_shuffle_shape(self):
        from localization_lens.modeling import PixelShuffleTokens

        tokens = torch.arange(1 * 64 * 3).reshape(1, 64, 3).float()
        output = PixelShuffleTokens(4)(tokens, 8, 8)
        self.assertEqual(tuple(output.shape), (1, 4, 48))

    def test_pool_semantic_views_handles_variable_tile_counts(self):
        from localization_lens.modeling import pool_semantic_views

        # Sample 0 has one tile/view; sample 1 has two tiles/view. The model
        # removes the three zero-padding images from sample 0 before encoding.
        pixels = torch.ones(2, 6, 1, 1, 1)
        pixels[0, 3:] = 0
        features = torch.arange(9 * 2 * 4).reshape(9, 2, 4).float()
        pooled = pool_semantic_views(features, pixels)
        self.assertEqual(tuple(pooled.shape), (2, 3, 4))

    def test_dcl_is_finite(self):
        from localization_lens.modeling import DecoupledContrastiveLoss

        first = torch.randn(4, 8, requires_grad=True)
        second = first.detach() + 0.01 * torch.randn(4, 8)
        loss = DecoupledContrastiveLoss()(first, second)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(first.grad)

    def test_dcl_accepts_mixed_precision_inputs(self):
        from localization_lens.modeling import DecoupledContrastiveLoss

        first = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
        second = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
        loss = DecoupledContrastiveLoss()(first, second)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()

    def test_infonce_is_finite(self):
        from localization_lens.modeling import InfoNCELoss

        images = torch.randn(4, 8, requires_grad=True)
        text = images.detach() + 0.01 * torch.randn(4, 8)
        loss = InfoNCELoss()(images, text)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(images.grad)


if __name__ == "__main__":
    unittest.main()
