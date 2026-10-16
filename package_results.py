
import json
from pathlib import Path

ROOT = Path(".")

def load(path):
    return json.loads(Path(path).read_text())

# Main classification experiments
mnist = load("MNIST/config.json")
cifar = load("Cifar10/config.json")
imagenet = load("Imagenet/config.json")

# Deep-SCM overlays (partial) + shared defaults used by turtle core
morpho = load("benchmark/morphomnist/graco_config.json")
celeba_s = load("benchmark/celeba/graco_config_simple.json")
celeba_c = load("benchmark/celeba/graco_config_complex.json")

# Defaults from benchmark/shared/graco_turtle_core.py (used when not in JSON)
BENCH_DEFAULTS = {
    "min_init_tree_depth": 7,
    "max_init_tree_depth": 20,
    "max_tree_depth": 35,
    "max_wraps": 5,
    "codon_size": 1024,
}

def row(name, evo, runtime, overlay=None):
    if overlay is None:
        overlay = {}
    merged = {**BENCH_DEFAULTS, **evo, **runtime, **overlay}
    keys = [
        "population_size",
        "max_generations",
        "p_crossover",
        "p_mutation",
        "min_init_tree_depth",
        "max_init_tree_depth",
        "max_tree_depth",
        "max_wraps",
        "codon_size",
        "turtle_delta_scale",
        "batch_size_eval",
    ]
    print(f"\n=== {name} ===")
    for k in keys:
        print(f"{k}: {merged.get(k)}")

row("MNIST", mnist["evolution"], mnist["runtime"])
row("CIFAR-10", cifar["evolution"], cifar["runtime"])
row("ImageNet-100", imagenet["evolution"], imagenet["runtime"])
row("MorphoMNIST", {}, {}, morpho)
row("CelebA-simple", {}, {}, celeba_s)
row("CelebA-complex", {}, {}, celeba_c)

print("\nNOTE: MorphoMNIST/CelebA JSON only override a subset;")
print("depth/wrap/codon values come from graco_turtle_core.py defaults.")
