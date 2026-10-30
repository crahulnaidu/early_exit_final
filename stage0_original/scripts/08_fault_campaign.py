import sys
import json
import torch
import pandas as pd
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dataset.datamodule import get_cifar10_dataloaders, set_seed
from torchvision.models import vgg16_bn
from models.searched_exit_model import SearchedMultiExitVGG
from utils.exit_conditions import compute_energy_score
from faults.inject import inject_exponent_flips, undo_faults

def run_fault_campaign(mode="A", max_k=4, num_runs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================================================================", flush=True)
    print(f"  SECTION 6.3: FAULT CAMPAIGN (Mode={mode}, k=1..{max_k}, Runs={num_runs})", flush=True)
    print("==========================================================================================", flush=True)

    config_path = "checkpoints/chosen_config.json"
    calib_path = "checkpoints/calibrated_thresholds.json"

    with open(config_path, "r") as f:
        chosen_config = json.load(f)
    with open(calib_path, "r") as f:
        calib_data = json.load(f)

    positions = chosen_config["positions"]
    archs = chosen_config["archs"]
    calib = calib_data["alpha_1.0"]

    _, _, test_loader = get_cifar10_dataloaders(batch_size=128, seed=42)

    vgg_base = vgg16_bn(num_classes=10)
    vgg_base.load_state_dict(torch.load("checkpoints/vgg16_bn_cifar10_base.pth", map_location=device))

    model = SearchedMultiExitVGG(vgg_base.features, vgg_base.avgpool, vgg_base.classifier, positions, archs)
    model.load_state_dict(torch.load("checkpoints/ee_searched.pth", map_location=device), strict=False)
    model = model.to(device).eval()

    num_exits = len(positions)
    campaign_results = []

    for k in range(1, max_k + 1):
        for run in range(1, num_runs + 1):
            seed = 100000 * (ord(mode) - 64) + 1000 * k + run
            fault_log = inject_exponent_flips(model, mode=mode, k=k, seed=seed)

            correct = 0
            sdc = 0
            flagged = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    logits_list = model(inputs)
                    batch_size = targets.size(0)
                    total += batch_size

                    for i in range(batch_size):
                        exited = False
                        for ex_idx in range(num_exits):
                            logits = logits_list[ex_idx][i:i+1]
                            
                            if torch.isnan(logits).any() or torch.isinf(logits).any():
                                flagged += 1
                                exited = True
                                break

                            energy = compute_energy_score(logits).item()
                            tau_k = calib[f"tau_{ex_idx+1}"]
                            L_k = calib[f"L_{ex_idx+1}"]

                            if L_k <= energy <= tau_k:
                                exited = True
                                pred = logits.argmax(dim=-1).item()
                                if pred == targets[i].item():
                                    correct += 1
                                else:
                                    sdc += 1
                                break

                        if not exited:
                            final_logits = logits_list[-1][i:i+1]
                            if torch.isnan(final_logits).any() or torch.isinf(final_logits).any():
                                flagged += 1
                            else:
                                final_energy = compute_energy_score(final_logits).item()
                                if calib["L_fin"] <= final_energy <= calib["tau_fin"]:
                                    pred = final_logits.argmax(dim=-1).item()
                                    if pred == targets[i].item():
                                        correct += 1
                                    else:
                                        sdc += 1
                                else:
                                    flagged += 1

            undo_faults(model, fault_log)

            acc = 100.0 * correct / total
            sdc_rate = 100.0 * sdc / total
            flag_rate = 100.0 * flagged / total

            campaign_results.append({
                "mode": mode,
                "k": k,
                "run": run,
                "accuracy": acc,
                "sdc_rate": sdc_rate,
                "flag_rate": flag_rate
            })

            if run % 5 == 0 or run == 1:
                print(f"[Mode {mode} | k={k} | Run {run:2d}/{num_runs}] Acc: {acc:.2f}% | SDC Rate: {sdc_rate:.2f}% | Flag Rate: {flag_rate:.2f}%", flush=True)

    df = pd.DataFrame(campaign_results)
    out_file = f"checkpoints/fault_campaign_mode_{mode}.csv"
    df.to_csv(out_file, index=False)
    print(f"\n[Saved Campaign Results] {out_file}", flush=True)

def main():
    set_seed(42)
    run_fault_campaign(mode="A", max_k=4, num_runs=10)

if __name__ == "__main__":
    main()
