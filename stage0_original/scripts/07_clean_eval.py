import sys
import json
import torch
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dataset.datamodule import get_cifar10_dataloaders, set_seed
from torchvision.models import vgg16_bn
from models.searched_exit_model import SearchedMultiExitVGG
from utils.exit_conditions import compute_energy_score
from utils.flops import TOTAL_BASE_FLOPS

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = "checkpoints/chosen_config.json"
    calib_path = "checkpoints/calibrated_thresholds.json"

    with open(config_path, "r") as f:
        chosen_config = json.load(f)
    with open(calib_path, "r") as f:
        calib_data = json.load(f)

    positions = chosen_config["positions"]
    archs = chosen_config["archs"]
    calib = calib_data["alpha_1.0"] # Default evaluation at alpha = 1.0%

    _, _, test_loader = get_cifar10_dataloaders(batch_size=128, seed=42)

    vgg_base = vgg16_bn(num_classes=10)
    vgg_base.load_state_dict(torch.load("checkpoints/vgg16_bn_cifar10_base.pth", map_location=device))

    model = SearchedMultiExitVGG(vgg_base.features, vgg_base.avgpool, vgg_base.classifier, positions, archs)
    model.load_state_dict(torch.load("checkpoints/ee_searched.pth", map_location=device), strict=False)
    model = model.to(device).eval()

    num_exits = len(positions)
    exit_counts = [0] * (num_exits + 1)
    flagged_count = 0
    correct_count = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits_list = model(inputs)
            batch_size = targets.size(0)
            total_samples += batch_size

            for i in range(batch_size):
                exited = False
                for k in range(num_exits):
                    logits = logits_list[k][i:i+1]
                    energy = compute_energy_score(logits).item()
                    tau_k = calib[f"tau_{k+1}"]
                    L_k = calib[f"L_{k+1}"]

                    if L_k <= energy <= tau_k:
                        exited = True
                        exit_counts[k] += 1
                        pred = logits.argmax(dim=-1).item()
                        if pred == targets[i].item():
                            correct_count += 1
                        break

                if not exited:
                    # Final Exit
                    final_logits = logits_list[-1][i:i+1]
                    final_energy = compute_energy_score(final_logits).item()
                    if calib["L_fin"] <= final_energy <= calib["tau_fin"]:
                        exit_counts[-1] += 1
                        pred = final_logits.argmax(dim=-1).item()
                        if pred == targets[i].item():
                            correct_count += 1
                    else:
                        flagged_count += 1

    clean_acc = 100.0 * correct_count / total_samples
    far_pct = 100.0 * flagged_count / total_samples

    print("\n==========================================================================================", flush=True)
    print("  SECTION 7: CLEAN BENCHMARK EVALUATION (TEST SET)", flush=True)
    print("==========================================================================================", flush=True)
    print(f"Overall Clean Accuracy : {clean_acc:.2f}%", flush=True)
    print(f"Exit Distribution      : {exit_counts} (E1..EK, Final)", flush=True)
    print(f"Clean Flagged (FAR)    : {flagged_count} / {total_samples} ({far_pct:.2f}%)", flush=True)
    print("==========================================================================================", flush=True)

if __name__ == "__main__":
    main()
