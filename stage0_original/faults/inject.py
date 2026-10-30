import torch
import random

def inject_exponent_flips(model, mode="A", k=1, seed=42):
    """
    Injects vectorized FP32 exponent bit flips into Conv2d and Linear weight tensors.
    Bits 23..30 represent the FP32 exponent field (bit 30 = MSB).
    """
    random.seed(seed)
    torch.manual_seed(seed)

    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            target_modules.append((name, module))

    if not target_modules:
        return []

    fault_log = []

    if mode == "A":
        # Mode A: k distinct weights, 1 random exponent bit (23..30) each
        for _ in range(k):
            name, module = random.choice(target_modules)
            weight = module.weight.data
            flat_idx = random.randint(0, weight.numel() - 1)
            bit = random.randint(23, 30)

            w_int = weight.view(-1).view(torch.int32)
            old_val = weight.view(-1)[flat_idx].item()
            w_int[flat_idx] ^= (1 << bit)
            new_val = weight.view(-1)[flat_idx].item()

            fault_log.append((name, flat_idx, bit, old_val, new_val))

    elif mode == "B":
        # Mode B: 1 random weight, k distinct exponent bits flipped
        name, module = random.choice(target_modules)
        weight = module.weight.data
        flat_idx = random.randint(0, weight.numel() - 1)
        bits = random.sample(range(23, 31), k)

        w_int = weight.view(-1).view(torch.int32)
        old_val = weight.view(-1)[flat_idx].item()
        for bit in bits:
            w_int[flat_idx] ^= (1 << bit)
        new_val = weight.view(-1)[flat_idx].item()

        fault_log.append((name, flat_idx, bits, old_val, new_val))

    elif mode == "C":
        # Mode C (MENDNet Sign Flip): k distinct weights, bit 31 flipped
        for _ in range(k):
            name, module = random.choice(target_modules)
            weight = module.weight.data
            flat_idx = random.randint(0, weight.numel() - 1)
            bit = 31

            w_int = weight.view(-1).view(torch.int32)
            old_val = weight.view(-1)[flat_idx].item()
            w_int[flat_idx] ^= (1 << bit)
            new_val = weight.view(-1)[flat_idx].item()

            fault_log.append((name, flat_idx, bit, old_val, new_val))

    return fault_log

def undo_faults(model, fault_log):
    """
    Restores original weights exactly by reapplying XOR masks on logged indices.
    """
    target_modules = {name: module for name, module in model.named_modules()}

    for item in fault_log:
        name, flat_idx, bit_or_bits, _, _ = item
        module = target_modules[name]
        weight = module.weight.data
        w_int = weight.view(-1).view(torch.int32)

        if isinstance(bit_or_bits, list):
            for bit in bit_or_bits:
                w_int[flat_idx] ^= (1 << bit)
        else:
            w_int[flat_idx] ^= (1 << bit_or_bits)
