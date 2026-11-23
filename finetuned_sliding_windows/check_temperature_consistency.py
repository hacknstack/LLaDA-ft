"""CPU toy-decoder audit; no model weights or external data required."""
import importlib.util
import itertools
import json
import math
import statistics
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch

torch.set_num_threads(2)
folder = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).parent
sys.path.insert(0, str(folder))
spec = importlib.util.spec_from_file_location('extraction_under_test', folder / 'probabilistic_extraction.py')
pe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pe
spec.loader.exec_module(pe)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter('anchor', torch.nn.Parameter(torch.zeros((), dtype=torch.float64)))

    def forward(self, ids, attention_mask=None):
        # Identical initial distributions exercise smallest-index tie handling;
        # revealed targets change the remaining position's distribution.
        logits = torch.zeros((*ids.shape, 3), dtype=torch.float64)
        logits[:, 0] = torch.tensor([1.2, 0.4, -0.8], dtype=torch.float64)
        logits[:, 1] = torch.tensor([1.2, 0.4, -0.8], dtype=torch.float64)
        logits[ids[:, 1] != 3, 0] = torch.tensor([0.1, 1.0, -0.5], dtype=torch.float64)
        logits[ids[:, 0] != 3, 1] = torch.tensor([-0.6, 1.4, 0.5], dtype=torch.float64)
        return SimpleNamespace(logits=logits)


model = Toy()
sequence = torch.zeros((1, 100), dtype=torch.long)
sequence[0, 1] = 1
targets = [0, 1]


def softmax(values, temperature):
    maximum = max(values)
    weights = [math.exp((v - maximum) / temperature) for v in values]
    return [v / sum(weights) for v in weights]


def brute_force(temperature, scheme, threshold):
    @lru_cache(None)
    def recurse(revealed):
        if len(revealed) == 2:
            return 1.0
        x = sequence.clone()
        active = [i for i in range(2) if i not in revealed]
        x[0, active] = 3
        logits = model(x).logits[0].tolist()
        probs = {i: softmax(logits[i], temperature) for i in active}
        conf = {i: softmax(logits[i], 1.0) for i in active}
        result = 0.0
        for choices in itertools.product(range(3), repeat=len(active)):
            tokens = dict(zip(active, choices))
            mass = math.prod(probs[i][tokens[i]] for i in active)
            selected = ([i for i in active if conf[i][tokens[i]] >= threshold]
                        if scheme == 'threshold' else [])
            if not selected:
                selected = [max(active, key=lambda i: (conf[i][tokens[i]], -i))]
            if all(tokens[i] == targets[i] for i in selected):
                result += mass * recurse(tuple(sorted((*revealed, *selected))))
        return result
    return recurse(())


report = []
for temperature in (0.5, 1.0, 2.0):
    for scheme, threshold in [('low-confidence', 0.9), ('threshold', 0.5), ('threshold', 0.9)]:
        expected = brute_force(temperature, scheme, threshold)
        common = dict(model=model, sequence_tokens=sequence, masked_indexes=[1, 2],
                      steps=2, attention_mask=None, mask_id=3, temperature=temperature)
        public = dict(model=model, prompt_tokens=sequence[:, :50], target_tokens=sequence[:, 50:],
                      steps=2, mask_id=3, masked_indexes=[1, 2], temperature=temperature,
                      remasking=scheme, confidence_threshold=threshold)
        exact = pe.compute_diffusion_probabilistic_extraction(**public, estimation_method='exact')
        assert math.isclose(exact['probability'], expected, rel_tol=1e-10, abs_tol=1e-12)
        path = pe.compute_diffusion_probabilistic_extraction(
            **public, estimation_method='path_sampling', num_samples=4000, seed=42, verbose=True)
        if scheme == 'threshold':
            mc = pe._monte_carlo_threshold_probability_fast_from_partially_masked(
                **common, confidence_threshold=threshold, num_samples=80000, seed=43)
        else:
            mc = pe._monte_carlo_probability_temperature_fast_from_partially_masked(
                **common, decoding_scheme='full', k=1, num_samples=80000, seed=43)
        path_se = statistics.stdev(path['sample_probabilities']) / math.sqrt(4000)
        assert abs(path['probability'] - expected) <= 7 * path_se + 1e-12
        mc_se = math.sqrt(expected * (1 - expected) / 80000)
        assert abs(mc.estimate - expected) <= 7 * mc_se
        report.append(dict(temperature=temperature, scheme=scheme, threshold=threshold,
                           enumerated=expected, exact=exact['probability'],
                           path=path['probability'], monte_carlo=mc.estimate))

for temperature in (0.0, -1.0, float('inf'), float('nan')):
    try:
        pe.compute_diffusion_probabilistic_extraction(
            model=model, prompt_tokens=sequence[:, :50], target_tokens=sequence[:, 50:],
            steps=2, mask_id=3, masked_indexes=[1, 2], temperature=temperature,
            remasking='low-confidence', estimation_method='path_sampling')
    except ValueError as error:
        assert 'temperature' in str(error)
    else:
        raise AssertionError('Invalid temperature accepted')

print(json.dumps(dict(source=str(folder), checks=report, invalid_temperatures_rejected=True), indent=2))
