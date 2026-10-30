import sys
import json
import torch
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dataset.datamodule import get_cifar10_dataloaders, set_seed
from torchvision.models import vgg16_bn
from models.searched_exit_model import SearchedMultiExitVGG
from utils.exit_conditions import compute_energy_score

def calibrate_thresholds_for_alpha(val_energies_list, val_correct_list, target_alpha_pct):
    """
    Calibrates tau_k, L_k, and tau_fin to achieve target clean False Alarm Rate (alpha %).
    """
    K = len(val_energies_list) - 1
    calib = {}

    for k in range(K):
        energies = val_energies_list[k]
        # Lower bound L_k = 0.1 percentile of clean validation energy
        L_k = float(np.percentile(energies, 0.1))
        # Upper bound tau_k chosen at 70th percentile for early exits
        tau_k = float(np.percentile(energies, 70.0))
        calib[f"L_{k+1}"] = L_k
        calib[f"tau_{k+1}"] = tau_k

    # Final Exit Threshold (tau_fin) calibrated to enforce target FAR = alpha %
    final_energies = val_energies_list[-1]
    tau_fin = float(np.percentile(final_energies, 100.0 - target_alpha_pct))
    L_fin = float(np.percentile(final_energies, 0.1))

    calib["L_fin"] = L_fin
    calib["tau_fin"] = tau_fin
    calib["alpha_pct"] = target_alpha_pct

    return calib

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Section 5.5] Calibrating Energy Thresholds on Clean Validation Set ({device})", flush=True)

    config_path = "checkpoints/chosen_config.json"
    if not Path(config_path).exists():
        print(f"[Error] Configuration file {config_path} not found!", flush=True)
        return

    with open(config_path, "r") as f:
        chosen_config = json.load(f)

    positions = chosen_config["positions"]
    archs = chosen_config["archs"]

    _, val_loader, _ = get_cifar10_dataloaders(batch_size=128, seed=42)

    vgg_base = vgg16_bn(num_classes=10)
    base_ckpt = "checkpoints/vgg16_bn_cifar10_base.pth"
    ee_ckpt = "checkpoints/ee_searched.pth"

    vgg_base.load_state_dict(torch.load(base_ckpt, map_location=device))
    model = SearchedMultiExitVGG(vgg_base.features, vgg_base.avgpool, vgg_base.classifier, positions, archs)
    model.load_state_dict(torch.load(ee_ckpt, map_location=device), strict=False)
    model = model.to(device).eval()

    val_energies_list = [[] for _ in range(len(positions) + 1)]
    val_correct_list = [[] for _ in range(len(positions) + 1)]

    print("\n--> Extracting clean validation energy distributions ...", flush=True)
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits_list = model(inputs)

            for k, logits in enumerate(logits_list):
                energies = compute_energy_score(logits)
                preds = logits.argmax(dim=-1)
                corrects = preds.eq(targets)

                val_energies_list[k].extend(energies.cpu().numpy().tolist())
                val_correct_list[k].extend(corrects.cpu().numpy().tolist())

    # Sweep alpha targets: 0.5%, 1.0%, 2.0%, 5.0%
    alpha_targets = [0.5, 1.0, 2.0, 5.0]
    calibration_results = {}

    print("\n==========================================================================================", flush=True)
    print("  SECTION 5.5: ENERGY THRESHOLD CALIBRATION RESULTS", flush=True)
    print("==========================================================================================", flush=True)

    for alpha in alpha_targets:
        calib = calibrate_thresholds_for_alpha(val_energies_list, val_correct_list, alpha)
        calibration_results[f"alpha_{alpha}"] = calib
        print(f"[Alpha = {alpha}%] tau_fin: {calib['tau_fin']:.4f} | L_fin: {calib['L_fin']:.4f}", flush=True)

    out_file = "checkpoints/calibrated_thresholds.json"
    with open(out_file, "w") as f:
        json.dump(calibration_results, f, indent=4)

    print(f"\n[Saved Calibrated Thresholds] {out_file}", flush=True)
    print("==========================================================================================", flush=True)

if __name__ == "__main__":
    main()
