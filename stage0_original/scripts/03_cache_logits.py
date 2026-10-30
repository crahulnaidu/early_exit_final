import sys
import torch
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dataset.datamodule import get_cifar10_dataloaders, set_seed
from torchvision.models import vgg16_bn
from models.multi_exit import SupernetMultiExitVGG

def cache_split_logits(supernet, dataloader, split_name, device):
    print(f"--> Caching logits for split: {split_name} ...", flush=True)
    supernet.eval()

    all_logits = {k: [] for k in supernet.head_keys}
    all_logits["final"] = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            logits_dict = supernet(inputs)

            for key in all_logits.keys():
                all_logits[key].append(logits_dict[key].cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    cached_data = {
        "targets": torch.cat(all_targets, dim=0),
        "logits": {k: torch.cat(v, dim=0) for k, v in all_logits.items()}
    }

    out_dir = Path("checkpoints/cached_logits")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{split_name}_cached_logits.pt"
    
    torch.save(cached_data, out_file)
    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"    Saved {split_name} logits to {out_file} ({size_mb:.2f} MB)", flush=True)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Section 2.3] Caching Candidate Logits on Device: {device}", flush=True)

    _, val_loader, test_loader = get_cifar10_dataloaders(batch_size=128, seed=42)

    # Instantiate model structure
    vgg_base = vgg16_bn(num_classes=10)
    supernet = SupernetMultiExitVGG(
        features=vgg_base.features,
        avgpool=vgg_base.avgpool,
        classifier=vgg_base.classifier
    )

    # Load trained weights
    base_ckpt = "checkpoints/vgg16_bn_cifar10_base.pth"
    supernet_ckpt = "checkpoints/supernet_heads.pth"

    if Path(base_ckpt).exists():
        vgg_base.load_state_dict(torch.load(base_ckpt, map_location=device))
        print(f"[Loaded Base Backbone] {base_ckpt}", flush=True)

    if Path(supernet_ckpt).exists():
        supernet.load_state_dict(torch.load(supernet_ckpt, map_location=device), strict=False)
        print(f"[Loaded Supernet Heads] {supernet_ckpt}", flush=True)

    supernet = supernet.to(device)

    print("\n==========================================================================================", flush=True)
    print("  SECTION 2.3: CACHING LOGITS FOR FAST NAS SEARCH", flush=True)
    print("==========================================================================================", flush=True)

    cache_split_logits(supernet, val_loader, "val", device)
    cache_split_logits(supernet, test_loader, "test", device)

    print("\n==========================================================================================", flush=True)
    print("  LOGIT CACHING COMPLETE", flush=True)
    print("==========================================================================================", flush=True)

if __name__ == "__main__":
    main()
