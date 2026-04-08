import argparse
import json
import os

import torch
from tqdm import tqdm

import config
from task1.dataset import get_segmentation_loaders
from task1.models import UNet, AEEncoder, AESegmentationModel
from task1.dataset import TissueSegmentationDataset
from task1.utils import (
    set_seed, get_device,
    SegmentationMetrics,
    save_prediction_grid, save_confusion_matrix,
    save_class_distribution, save_model_comparison_grid,
    save_metrics_comparison_chart,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate segmentation models")
    p.add_argument("--unet_ckpt", type=str,
                    default=os.path.join(config.CHECKPOINT_DIR,
                                          "unet_best.pth"))
    p.add_argument("--ae_seg_ckpt", type=str,
                    default=os.path.join(config.CHECKPOINT_DIR,
                                          "ae_seg_best.pth"))
    p.add_argument("--batch_size", type=int, default=config.UNET_BATCH_SIZE)
    p.add_argument("--patch_size", type=int, default=config.PATCH_SIZE)
    p.add_argument("--data_root", type=str, default=config.DATASET_ROOT)
    p.add_argument("--results_dir", type=str, default=config.RESULTS_DIR)
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


@torch.no_grad()
def evaluate_model(model, test_loader, device, name: str, results_dir: str):
    """Run a model on the test set and return metrics."""
    model.eval()
    metrics = SegmentationMetrics(device=device)
    saved_grid = False

    for images, masks in tqdm(test_loader, desc=f"  Eval {name}"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        metrics.update(logits, masks)

        if not saved_grid:
            preds = logits.argmax(dim=1)
            save_prediction_grid(
                images, masks, preds,
                os.path.join(results_dir, f"{name}_test_predictions.png"),
                num_samples=6,
            )
            saved_grid = True

    # Confusion matrix
    save_confusion_matrix(
        metrics,
        os.path.join(results_dir, f"{name}_confusion_matrix.png"))

    return metrics


def load_unet(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    features = args.get("features", config.UNET_FEATURES)
    model = UNet(features=features).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def load_ae_seg(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    features = ckpt.get("features", config.AE_FEATURES)
    encoder = AEEncoder(features=features)
    model = AESegmentationModel(encoder, freeze_encoder=True,
                                 features=features).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def main():
    args = parse_args()

    if args.data_root != config.DATASET_ROOT:
        config.DATASET_ROOT = args.data_root
        config.TRAIN_IMAGE_DIR = os.path.join(args.data_root, "train", "image")
        config.TRAIN_LABEL_DIR = os.path.join(args.data_root, "train", "tissue")
        config.VAL_IMAGE_DIR = os.path.join(args.data_root, "validation", "image")
        config.VAL_LABEL_DIR = os.path.join(args.data_root, "validation", "tissue")
        config.TEST_IMAGE_DIR = os.path.join(args.data_root, "test", "image")
        config.TEST_LABEL_DIR = os.path.join(args.data_root, "test", "tissue")

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)

    # Test loader
    _, _, test_loader = get_segmentation_loaders(
        batch_size=args.batch_size, patch_size=args.patch_size)

    # Evaluate models
    all_results = {}

    # UNet
    if os.path.exists(args.unet_ckpt):
        print(f"\n{'='*55}")
        print("  Evaluating UNet on test set")
        print(f"{'='*55}")
        model = load_unet(args.unet_ckpt, device)
        print(f"  Trainable params: {model.count_parameters():,}")
        metrics = evaluate_model(model, test_loader, device,
                                  "unet", args.results_dir)
        metrics.print_summary()
        all_results["UNet"] = metrics.summary()
        all_results["UNet"]["trainable_params"] = model.count_parameters()
        del model
    else:
        print(f"UNet checkpoint not found: {args.unet_ckpt}")

    # AE-Seg
    if os.path.exists(args.ae_seg_ckpt):
        print(f"\n{'='*55}")
        print("  Evaluating AE-Seg on test set")
        print(f"{'='*55}")
        model = load_ae_seg(args.ae_seg_ckpt, device)
        trainable = model.count_parameters(True)
        total = model.count_parameters(False)
        print(f"  Trainable params (decoder): {trainable:,}")
        print(f"  Total params:               {total:,}")
        metrics = evaluate_model(model, test_loader, device,
                                  "ae_seg", args.results_dir)
        metrics.print_summary()
        all_results["AE-Seg"] = metrics.summary()
        all_results["AE-Seg"]["trainable_params_decoder"] = trainable
        all_results["AE-Seg"]["total_params"] = total
        del model
    else:
        print(f"AE-Seg checkpoint not found: {args.ae_seg_ckpt}")

    # Comparison table
    if len(all_results) >= 2:
        print(f"\n{'='*70}")
        print("  COMPARISON TABLE")
        print(f"{'='*70}")
        header = f"  {'Metric':<25}"
        for name in all_results:
            header += f" {name:>12}"
        print(header)
        print("  " + "-" * (25 + 13 * len(all_results)))

        compare_keys = ["pixel_accuracy", "mean_dice", "mean_iou"]
        for cname in config.CLASS_NAMES:
            compare_keys += [f"dice_{cname}", f"iou_{cname}"]

        for key in compare_keys:
            row = f"  {key:<25}"
            for name in all_results:
                val = all_results[name].get(key, -1)
                row += f" {val:>12.4f}"
            print(row)
        print(f"{'='*70}")

    # Save results to JSON
    results_path = os.path.join(args.results_dir, "test_results.json")
    # Convert numpy values to Python floats for JSON serialisation
    json_results = {}
    for name, res in all_results.items():
        json_results[name] = {k: float(v) if hasattr(v, 'item') else v
                              for k, v in res.items()}
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nResults saved -> {results_path}")

    # Side-by-side model comparison (2.2.2b)
    if len(all_results) >= 2:
        print("\nGenerating side-by-side model comparison...")
        models = {}
        if os.path.exists(args.unet_ckpt):
            models["UNet"] = load_unet(args.unet_ckpt, device)
        if os.path.exists(args.ae_seg_ckpt):
            models["AE-Seg"] = load_ae_seg(args.ae_seg_ckpt, device)

        for m in models.values():
            m.eval()

        print("Finding best and worst predictions for extreme comparison...")
        best_score = -1.0
        best_data = None
        worst_score = 2.0
        worst_data = None
        second_worst_score = 2.0
        second_worst_data = None
        saved_first_batch = False

        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="  Generative Visualisations"):
                images = images.to(device)
                masks = masks.to(device)
                model_preds = {
                    name: m(images).argmax(dim=1)
                    for name, m in models.items()
                }

                if not saved_first_batch:
                    save_model_comparison_grid(
                        images, masks, model_preds,
                        os.path.join(args.results_dir,
                                     "model_comparison.png"),
                        num_samples=6,
                    )
                    saved_first_batch = True

                # Evaluate per image to find extremes
                for i in range(images.size(0)):
                    img_i = images[i:i+1] # keep batch dim
                    mask_i = masks[i:i+1] # keep batch dim
                    preds_i = {name: p[i:i+1] for name, p in model_preds.items()}
                    
                    # Calculate unique classes for the top image condition
                    num_unique_classes = len(torch.unique(mask_i))
                    has_tissue = (mask_i > 0).any()
                    
                    if not has_tissue:
                        continue
                        
                    avg_image_dice = 0.0
                    for name, p_i in preds_i.items():
                        model_dice = 0.0
                        present_classes = 0
                        for c in range(config.NUM_CLASSES):
                            gt_c = (mask_i == c)
                            if not gt_c.any():
                                continue
                            pred_c = (p_i == c)
                            tp = (pred_c & gt_c).sum().float()
                            fp = (pred_c & ~gt_c).sum().float()
                            fn = (~pred_c & gt_c).sum().float()
                            denom = 2.0 * tp + fp + fn
                            dice_c = (2.0 * tp / denom) if denom > 0 else 0.0
                            model_dice += dice_c
                            present_classes += 1
                        avg_image_dice += (model_dice / present_classes).item()
                        
                    avg_image_dice /= len(preds_i)
                    
                    # For top image: must have more than 1 class
                    if num_unique_classes > 1 and avg_image_dice > best_score:
                        best_score = avg_image_dice
                        best_data = (img_i, mask_i, preds_i)
                        
                    # For bottom image: track worst and second worst
                    if avg_image_dice < worst_score:
                        # old worst becomes second worst
                        second_worst_score = worst_score
                        second_worst_data = worst_data
                        
                        worst_score = avg_image_dice
                        worst_data = (img_i, mask_i, preds_i)
                    elif avg_image_dice < second_worst_score:
                        second_worst_score = avg_image_dice
                        second_worst_data = (img_i, mask_i, preds_i)

        final_worst_data = second_worst_data if second_worst_data is not None else worst_data

        if best_data is not None and final_worst_data is not None:
            print(f"  Selected top image (score {best_score:.4f}) and bottom image (score {second_worst_score:.4f}, excluded abs worst {worst_score:.4f})")
            
            ext_images = torch.cat([best_data[0], final_worst_data[0]], dim=0)
            ext_masks = torch.cat([best_data[1], final_worst_data[1]], dim=0)
            ext_preds = {name: torch.cat([best_data[2][name], final_worst_data[2][name]], dim=0) for name in models.keys()}
            
            save_model_comparison_grid(
                ext_images, ext_masks, ext_preds,
                os.path.join(args.results_dir, "extreme_model_comparison.png"),
                num_samples=2,
            )

        save_metrics_comparison_chart(
            all_results,
            os.path.join(args.results_dir, "metrics_comparison.png"))

        del models

    # Class distribution analysis (2.2.2c)
    print("\nAnalysing class distribution...")
    train_ds = TissueSegmentationDataset(
        config.TRAIN_IMAGE_DIR, config.TRAIN_LABEL_DIR,
        patch_size=args.patch_size, is_train=False)
    class_dist = save_class_distribution(
        train_ds,
        os.path.join(args.results_dir, "class_distribution.png"))

    # Baseline comparison
    print("\n  BASELINE COMPARISON")
    print("  " + "-" * 50)
    print(f"  Baseline mean Dice: {config.BASELINE_DICE:.4f}")
    print(f"  Baseline params:    {config.BASELINE_PARAMS:,}")
    for name, res in all_results.items():
        md = res.get("mean_dice", 0)
        delta = md - config.BASELINE_DICE
        print(f"  {name} mean Dice: {md:.4f}  "
              f"(delta = {delta:+.4f})")
    print()


if __name__ == "__main__":
    main()