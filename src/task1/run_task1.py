import argparse
import os
import subprocess
import sys
import config


def run(cmd: str):
    """Run a shell command, forwarding stdout/stderr."""
    print(f"\n{'#' * 70}")
    print(f"# RUNNING: {cmd}")
    print(f"{'#' * 70}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    p = argparse.ArgumentParser(description="Run full Task 1 pipeline")
    p.add_argument("--data_root", type=str, default=config.DATASET_ROOT,
                    help="Root directory of the Puma dataset")
    p.add_argument("--patch_size", type=int, default=config.PATCH_SIZE)
    p.add_argument("--unet_epochs", type=int, default=config.UNET_EPOCHS)
    p.add_argument("--ae_epochs", type=int, default=config.AE_EPOCHS)
    p.add_argument("--ae_seg_epochs", type=int, default=config.AE_SEG_EPOCHS)
    p.add_argument("--unet_batch_size", type=int, default=config.UNET_BATCH_SIZE)
    p.add_argument("--ae_batch_size", type=int, default=config.AE_BATCH_SIZE)
    p.add_argument("--ae_seg_batch_size", type=int, default=config.AE_SEG_BATCH_SIZE)
    p.add_argument("--use_class_weights", action="store_true",
                    help="Use inverse‑frequency class weights for CE loss")
    p.add_argument("--lambda_dice", type=float, default=config.LAMBDA_DICE)
    p.add_argument("--lambda_ce", type=float, default=config.LAMBDA_CE)
    p.add_argument("--skip_unet", action="store_true")
    p.add_argument("--skip_ae", action="store_true")
    p.add_argument("--skip_ae_seg", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    args = p.parse_args()

    cw_flag = "--use_class_weights" if args.use_class_weights else ""
    loss_flags = f"--lambda_dice {args.lambda_dice} --lambda_ce {args.lambda_ce}"
    python = sys.executable

    # ── Step 1: UNet ─────────────────────────
    if not args.skip_unet:
        run(f"{python} -m task1.train_unet "
            f"--data_root {args.data_root} "
            f"--epochs {args.unet_epochs} "
            f"--batch_size {args.unet_batch_size} "
            f"--patch_size {args.patch_size} "
            f"{cw_flag} {loss_flags}")

    # ── Step 2: Autoencoder pre‑training ─────
    if not args.skip_ae:
        run(f"{python} -m task1.train_autoencoder "
            f"--data_root {args.data_root} "
            f"--epochs {args.ae_epochs} "
            f"--batch_size {args.ae_batch_size} "
            f"--patch_size {args.patch_size}")

    # ── Step 3: AE‑Seg ───────────────────────
    if not args.skip_ae_seg:
        run(f"{python} -m task1.train_ae_seg "
            f"--data_root {args.data_root} "
            f"--epochs {args.ae_seg_epochs} "
            f"--batch_size {args.ae_seg_batch_size} "
            f"--patch_size {args.patch_size} "
            f"{cw_flag} {loss_flags}")

    # ── Step 4: Evaluate ─────────────────────
    if not args.skip_eval:
        run(f"{python} -m task1.evaluate "
            f"--data_root {args.data_root} "
            f"--batch_size {args.unet_batch_size} "
            f"--patch_size {args.patch_size}")

    print("\n" + "=" * 70)
    print("  ALL STEPS COMPLETE")
    print("=" * 70)
    print(f"  Checkpoints: ./checkpoints/")
    print(f"  Results:     ./results/")
    print()


if __name__ == "__main__":
    main()