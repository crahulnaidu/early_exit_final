# VGG16-BN CIFAR-10 layer-wise fault sensitivity

## Scope

This experiment extends the existing grouped campaign with one persistent FP32 weight-bit fault in one discovered \`Conv2d\` or \`Linear\` module per realization. Exponent faults draw one bit uniformly from 23--30; sign faults flip bit 31. The code rejects a candidate if the selected weight or any output logit becomes non-finite, undoes it, and draws another candidate. Rejected attempts are logged. This preserves the existing campaign convention; report rejection counts because it can under-sample particularly destructive bits.

The default layer scope includes convolutional and linear modules in the backbone, final classifier, and selected exit heads. BatchNorm scale/bias and all biases are excluded. The layer names in each CSV are the source of truth.

## Phase 0: verify environment/checkpoint

Run from the repository's \`stage2_paper\` directory on the configured remote environment. Check the checkpoint path before starting. Do not run full campaigns with \`--quick\`.

\`\`\`bash
cd /home1/jtt/early_exit_final/stage2_paper
test -f checkpoints/ee_searched.pth
test -f /home1/jtt/early_exit_final/stage1_base_nas/checkpoints/vgg16_bn_cifar10_base.pth
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
\`\`\`

## Phase 1: pilot / smoke test

\`\`\`bash
python scripts/43_layerwise_campaign.py \
  --ee-ckpt checkpoints/ee_searched.pth --tag layerwise_pilot \
  --split val --runs 2 --quick
python scripts/43_layerwise_campaign.py \
  --ee-ckpt checkpoints/ee_searched.pth --tag layerwise_pilot \
  --split test --runs 2 --quick
\`\`\`

Quick mode truncates the split and caps runs at two. Use it only to validate imports, checkpoint loading, hooks, shapes, file writes, and exact weight restoration. The quick results are not scientific results.

## Phase 2: full campaign

\`\`\`bash
python scripts/43_layerwise_campaign.py \
  --ee-ckpt checkpoints/ee_searched.pth --tag nas_s11_t229 \
  --split val --runs 100
python scripts/43_layerwise_campaign.py \
  --ee-ckpt checkpoints/ee_searched.pth --tag nas_s11_t229 \
  --split test --runs 100
\`\`\`

The two commands save validation and test logits separately. Outputs are under \`results/layerwise/nas_s11_t229/{val,test}/\`. Each layer/mechanism has an \`.npy\` array of shape \`[runs, samples, exits+final, classes]\` and a matching CSV containing layer, bit, tensor index, old/new value, attempts, rejected attempts, and undo status. \`status.json\` records the layer inventory and global weight checksum.

Start with the pilot and confirm the storage estimate before the full test campaign. With 100 runs, all layers, both mechanisms, and all 10,000 test images, stored logits can require multiple GB. Ensure enough free disk space.

## Phase 3: offline detector-policy evaluation

\`\`\`bash
python scripts/44_layerwise_eval.py --tag nas_s11_t229
\`\`\`

This calibrates the policies on clean validation logits only and evaluates the test fault logits. Outputs:
- \`results/layerwise/nas_s11_t229/evaluation/layerwise_runs.csv.gz\`
- \`results/layerwise/nas_s11_t229/evaluation/layerwise_summary.csv\`
- \`results/layerwise/nas_s11_t229/evaluation/status.json\`

The default comparison includes plain DNN, SDN-style max-softmax, the proposed margin-band policy, and clean-calibrated MENDNet. Add \`--include-faulty-val\` only for the explicitly separate MENDNet variant that tunes on faulted validation logits. This uses extra fault information and must not be presented as a clean-only baseline.

## Important comparison limitation

The current offline comparison runs policies on the same checkpoint, exit positions, and head weights. It isolates routing/detector differences; it is **not** an end-to-end reproduction of native SDN or MENDNet architectures. To report end-to-end results, run the campaign for each method's actual checkpoint/head placement using the same preprocessing, data split, fault policy, and calibration protocol, then combine those separately tagged results. Do not label a shared-head policy comparison as native end-to-end.

## Metric definitions

For each method, define its clean-correct subset on the clean **test** logits. On each fault realization:
- \`fault_accuracy_pct\`: accepted predictions correct over all test samples; flagged samples do not count as correct.
- \`mitigated_accuracy_pct\`: accuracy after the repository's fallback label is applied to flagged samples.
- \`sdc_rate_on_clean_correct_pct\`: silently wrong accepted predictions divided by the method's clean-correct test count.
- \`detection_ratio_on_clean_correct_pct\`: flags divided by (flags + SDC) on that same clean-correct subset.
- \`coverage_pct\`: fraction not flagged; \`flag_rate_pct\`: fraction flagged.

Aggregate uncertainty across fault realizations (not by treating all images under one persistent fault as independent runs). Report per-layer and per-mechanism results separately. Exponent and sign campaigns have different bit sampling policies and should not be pooled into one number.
