import torch
import torch.nn.functional as F

def compute_energy_score(logits, temperature=1.0):
    """
    Computes numerically stable Energy score: E(z) = -T * logsumexp(z / T)
    Lower (more negative) values mean higher model confidence.
    """
    max_z = torch.max(logits, dim=-1, keepdim=True).values
    scaled_logits = (logits - max_z) / temperature
    energy = -temperature * (max_z.squeeze(-1) + torch.log(torch.sum(torch.exp(scaled_logits), dim=-1)))
    return energy

def evaluate_two_sided_energy(logits, tau_k, L_k):
    """
    Two-Sided Energy Exit Condition:
    Exits iff L_k <= Energy(z) <= tau_k AND logits are finite.
    """
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        return False, None

    energy = compute_energy_score(logits)
    # Valid exit if energy lies within physical clean bounds
    valid = (energy >= L_k) & (energy <= tau_k)
    return valid.item(), energy.item()
