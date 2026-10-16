import torch
import numpy as np
from typing import Dict, List, Optional
from json import load
from importlib import import_module
from model import SCM
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, Subset
import os
import argparse

import sys
sys.path.append("../../")

from models.classifiers.classifier import Classifier
from models.classifiers.celeba_classifier import CelebaClassifier
from models.classifiers.celeba_complex_classifier import CelebaComplexClassifier
from models.classifiers.adni_classifier import ADNIClassifier
from datasets.morphomnist.dataset import MorphoMNISTLike
from datasets.celeba.dataset import Celeba
from datasets.adni.dataset import ADNI
from datasets.transforms import ReturnDictTransform, get_attribute_ids

from evaluation.metrics.minimality import minimality
from evaluation.embeddings.embeddings import get_embedding_model, get_embedding_fn
from evaluation.metrics.fid import fid
from evaluation.metrics.effectiveness import effectiveness
from evaluation.metrics.utils import save_plots
from datasets.morphomnist.dataset import unnormalize as unnormalize_morphomnist
from datasets.celeba.dataset import unnormalize as unnormalize_celeba
from datasets.adni.dataset import unnormalize as unnormalize_adni

try:
    from eval_interventions import load_intervention_manifest, manifest_intervention_tensors
    from eval_subset import load_eval_indices
except ImportError:
    load_intervention_manifest = None
    manifest_intervention_tensors = None
    load_eval_indices = None

torch.multiprocessing.set_sharing_strategy('file_system')

rng = np.random.default_rng()


def _load_benchmark_eval(full_test_set: Dataset):
    """Load optional fixed eval indices + intervention manifest from env."""
    indices = None
    manifest = None
    idx_path = os.environ.get("BENCHMARK_EVAL_INDICES")
    man_path = os.environ.get("BENCHMARK_EVAL_INTERVENTIONS")
    if idx_path and os.path.isfile(idx_path) and load_eval_indices is not None:
        indices = load_eval_indices(idx_path)
    if man_path and os.path.isfile(man_path) and load_intervention_manifest is not None:
        manifest = load_intervention_manifest(man_path)
    if indices is not None:
        return full_test_set, indices, manifest
    return full_test_set, None, manifest


def _factual_batch_at_index(test_set: Dataset, idx: int, device: str = "cuda") -> dict:
    batch = test_set[idx]
    if not isinstance(batch, dict):
        raise TypeError("Expected dict batch from ReturnDictTransform")
    out = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    if "image" in out and out["image"].ndim == 3:
        out["image"] = out["image"].unsqueeze(0)
    return out

dataclass_mapping = {
    "morphomnist": (MorphoMNISTLike, unnormalize_morphomnist),
    "celeba": (Celeba, unnormalize_celeba),
    "adni": (ADNI, unnormalize_adni)
}

def produce_qualitative_samples(dataset, scm, parents, intervention_source, unnormalize_fn, num=20, show_difference=False):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=7)

    produce_every = len(dataset) // num
    fig_idx = 0
    for i , batch in tqdm(enumerate(data_loader)):
        if i % produce_every == 0:
            res = [batch]

            for do_parent in parents:
                counterfactual = produce_counterfactuals(batch, scm, do_parent, intervention_source,
                                                         force_change=True, possible_values=dataset.possible_values)
                res.append(counterfactual)

            save_plots(res, fig_idx, parents, unnormalize_fn, show_difference=show_difference)
            fig_idx += 1
    return


def different_value(possible_values, value, bins, attribute):
    if bins is not None and attribute in bins:
        return np.digitize(possible_values, bins[attribute]) != np.searchsorted(bins[attribute], value)
    else:
        return possible_values != value


def produce_counterfactuals(factual_batch: torch.Tensor, scm: nn.Module, do_parent:str, intervention_source: Dataset,
                            force_change: bool = False, possible_values = None, device: str = 'cuda', bins = None,
                            fixed_interventions: Optional[dict] = None):
    factual_batch = {k: v.to(device) for k, v in factual_batch.items()}

    if fixed_interventions is not None:
        interventions = {k: v.to(device) for k, v in fixed_interventions.items()}
    elif force_change:
        possible_values = possible_values[do_parent]
        values = factual_batch[do_parent].cpu()
        if do_parent not in ["digit", "apoE", "slice"]:
            interventions = {do_parent: torch.cat([torch.tensor(np.random.choice(possible_values[different_value(possible_values, value, bins, do_parent)])).unsqueeze(0)
                                                for value in values]).view(-1).unsqueeze(1).to(device)}
        else:
            interventions = {do_parent: torch.cat([torch.tensor(rng.choice(possible_values[torch.where((different_value(possible_values, value, bins, do_parent)).any(dim=1))], axis=0)).unsqueeze(0)
                                                for value in values]).to(device)}
    else:
        batch_size, _ , _ , _ = factual_batch["image"].shape
        idxs = torch.randperm(len(intervention_source))[:batch_size]

        interventions = {do_parent: torch.cat([intervention_source[id][do_parent] for id in idxs]).view(-1).unsqueeze(1).to(device)
                        if do_parent not in ["digit", "apoE", "slice"] else torch.cat([intervention_source[id][do_parent].unsqueeze(0).to(device) for id in idxs])}

    with torch.no_grad():
        abducted_noise = scm.encode(**factual_batch)
        counterfactual_batch = scm.decode(interventions, **abducted_noise)

    return counterfactual_batch


def evaluate_effectiveness(test_set: Dataset, unnormalize_fn, batch_size:int , scm: nn.Module, attributes: List[str], do_parent:str,
                           intervention_source: Dataset, predictors: Dict[str, Classifier], dataset: str,
                           eval_indices: Optional[List[int]] = None, manifest: Optional[dict] = None):

    effectiveness_scores = {attr_key: [] for attr_key in attributes}

    if eval_indices is not None and manifest is not None:
        entries = [e for e in manifest["entries"] if e["do_parent"] == do_parent and e["sample_idx"] in set(eval_indices)]
        for entry in tqdm(entries, desc=f"effectiveness do({do_parent})"):
            factual_batch = _factual_batch_at_index(test_set, entry["sample_idx"])
            fixed = manifest_intervention_tensors(entry)
            counterfactuals = produce_counterfactuals(
                factual_batch, scm, do_parent, intervention_source,
                fixed_interventions=fixed,
            )
            e_score = effectiveness(counterfactuals, unnormalize_fn, predictors, dataset)
            for attr in attributes:
                effectiveness_scores[attr].append(e_score[attr])
    else:
        test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=7)
        for factual_batch in tqdm(test_data_loader):
            counterfactuals = produce_counterfactuals(factual_batch, scm, do_parent, intervention_source,
                                                      force_change=True, possible_values=test_set.possible_values, bins=test_set.bins)
            e_score = effectiveness(counterfactuals, unnormalize_fn, predictors, dataset)
            for attr in attributes:
                effectiveness_scores[attr].append(e_score[attr])

    effectiveness_score = {key: (round(np.mean(score), 3), round(np.std(score), 3)) for key, score in effectiveness_scores.items()}
    print(f"Effectiveness score do({do_parent}): {effectiveness_score}")
    return effectiveness_score


def _generate_aligned_counterfactuals(test_set, scm, attributes, eval_indices, manifest, intervention_source):
    """Generate one CF per (index, do_parent) using the intervention manifest."""
    index_set = set(eval_indices)
    images = []
    for entry in tqdm(manifest["entries"], desc="generate CFs"):
        if entry["sample_idx"] not in index_set:
            continue
        do_parent = entry["do_parent"]
        if do_parent not in attributes:
            continue
        factual_batch = _factual_batch_at_index(test_set, entry["sample_idx"])
        fixed = manifest_intervention_tensors(entry)
        cf_batch = produce_counterfactuals(
            factual_batch, scm, do_parent, intervention_source,
            fixed_interventions=fixed,
        )
        images.append(cf_batch["image"])
    return images


def evaluate_fid(real_set: Dataset, test_set: Dataset, batch_size: int, scm: nn.Module, attributes: List[str],
                 eval_indices: Optional[List[int]] = None, manifest: Optional[dict] = None):
    real_data_loader = torch.utils.data.DataLoader(real_set, batch_size=batch_size, shuffle=False)

    if eval_indices is not None and manifest is not None:
        counterfactual_images = _generate_aligned_counterfactuals(
            test_set, scm, attributes, eval_indices, manifest, real_set
        )
    else:
        test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        counterfactual_images = []
        for factual_batch in tqdm(test_data_loader):
            do_parent = attributes[0] if len(attributes) == 1 else attributes[len(counterfactual_images) % len(attributes)]
            counterfactual_batch = produce_counterfactuals(
                factual_batch, scm, do_parent, intervention_source=real_set,
                force_change=True, possible_values=test_set.possible_values, bins=getattr(real_set, "bins", None),
            )
            counterfactual_images.append(counterfactual_batch["image"])

    fid_score = fid(real_data_loader, counterfactual_images)
    print(f"FID: mean {round(float(np.mean(fid_score)), 3):.3f} std {round(float(np.std(fid_score)), 3):.3f}")
    return fid_score


def evaluate_minimality(real_set: Dataset, test_set: Dataset, batch_size: int, scm: nn.Module, attributes: List[str],
                        embedding: str = None, embedding_fn = None,
                        eval_indices: Optional[List[int]] = None, manifest: Optional[dict] = None):

    factuals = []
    counterfactuals = []
    interventions = []

    if eval_indices is not None and manifest is not None:
        index_set = set(eval_indices)
        for entry in tqdm(manifest["entries"], desc="minimality"):
            if entry["sample_idx"] not in index_set:
                continue
            do_parent = entry["do_parent"]
            if do_parent not in attributes:
                continue
            factual_batch = _factual_batch_at_index(test_set, entry["sample_idx"])
            fixed = manifest_intervention_tensors(entry)
            counterfactual_batch = produce_counterfactuals(
                factual_batch, scm, do_parent, real_set,
                fixed_interventions=fixed,
            )
            if embedding == "vae":
                factual_cond = torch.cat([factual_batch[att] for att in attributes], dim=1).to("cuda")
                counterfactual_cond = torch.cat([counterfactual_batch[att] for att in attributes], dim=1)
            elif "intensity" in factual_batch:
                factual_cond = factual_batch["intensity"].to("cuda")
                counterfactual_cond = counterfactual_batch["intensity"]
            else:
                factual_cond = None
                counterfactual_cond = None

            factual_features = embedding_fn(factual_batch["image"].to("cuda"), factual_cond)
            counterfactual_features = embedding_fn(counterfactual_batch["image"], counterfactual_cond)
            factuals.append((factual_features[0], factual_batch[do_parent].cpu().numpy()))
            counterfactuals.append((counterfactual_features[0], counterfactual_batch[do_parent].cpu().numpy()))
            interventions.append(do_parent)
    else:
        test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        for factual_batch in tqdm(test_data_loader):
            do_parent = attributes[0]
            counterfactual_batch = produce_counterfactuals(
                factual_batch, scm, do_parent, intervention_source=real_set,
                force_change=True, possible_values=test_set.possible_values, bins=getattr(real_set, "bins", None),
            )
            if embedding == "vae":
                factual_cond = torch.cat([factual_batch[att] for att in attributes], dim=1).to("cuda")
                counterfactual_cond = torch.cat([counterfactual_batch[att] for att in attributes], dim=1)
            elif "intensity" in factual_batch:
                factual_cond = factual_batch["intensity"].to("cuda")
                counterfactual_cond = counterfactual_batch["intensity"]
            else:
                factual_cond = None
                counterfactual_cond = None
            factual_features = embedding_fn(factual_batch["image"].to("cuda"), factual_cond)
            counterfactual_features = embedding_fn(counterfactual_batch["image"], counterfactual_cond)
            factuals.append((factual_features[0], factual_batch[do_parent].numpy()))
            counterfactuals.append((counterfactual_features[0], counterfactual_batch[do_parent].cpu().numpy()))
            interventions.append(do_parent)

    if not factuals:
        print("Minimality: no pairs to evaluate")
        return

    minimality_scores, prob1s, prob2s = minimality(
        real=factuals, generated=counterfactuals, interventions=interventions,
        bins=getattr(real_set, "bins", None), embedding=embedding,
    )
    print(f"Minimality score: mean {round(np.mean(minimality_scores), 3)}, std {round(np.std(minimality_scores), 3)}")
    print(f"Prob 1: {np.mean(prob1s)}, Prob 2: {np.mean(prob2s)}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, help="Config file for experiment.", default="./configs/adni/hvae.json")
    parser.add_argument("--classifier-config", '-clf', type=str, help="Classifier config file.", default="./configs/adni/classifier.json")
    parser.add_argument("--metrics", '-m',
                        nargs="+", type=str,
                        help="Metrics to calculate. "
                        "Choose one or more of [effectiveness, fid, minimality]. If not set, all metrics are calculated.",
                        choices=["composition", "effectiveness", "fid", "minimality"],
                        default=["effectiveness", "fid", "minimality"])
    parser.add_argument("--cycles", '-cc', nargs="+", type=int, help="Composition cycles.", default=[1, 10])
    parser.add_argument("--qualitative", '-qn', type=int, help="Number of qualitative results to produce", default=20)
    parser.add_argument("--show-difference", '-sd', action='store_true', help="Show counterfactual-factual difference on qualitative results")
    parser.add_argument("--embeddings", type=str, choices=["vgg", "clfs", "vae", "lpips", "clip"], help="What embeddings to use for composition metric. "
                        "Supported: [vgg, clfs, vae, lpips, clip]. If not set, will compute distance on image space")
    parser.add_argument("--sampling-temperature", '-temp', type=float, default=0.1, help="Sampling temperature, used for VAE, HVAE.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # torch.manual_seed(42)

    assert os.path.isfile(args.classifier_config), f"{args.classifier_config} is not a file"
    with open(args.classifier_config, 'r') as f:
        config_cls = load(f)

    assert os.path.isfile(args.config), f"{args.config} is not a file"
    with open(args.config, 'r') as f:
        config = load(f)

    dataset = config["dataset"]
    attribute_size = config["attribute_size"]

    models = {}
    for variable in config["causal_graph"].keys():
        if variable not in config["mechanism_models"]:
            continue
        model_config = config["mechanism_models"][variable]

        module = import_module(model_config["module"])
        model_class = getattr(module, model_config["model_class"])
        model = model_class(params=model_config["params"], attr_size=attribute_size)

        models[variable] = model
        if "finetune" in model_config["params"] and model_config["params"]["finetune"] == 1:
            model.name += '_finetuned'

    batch_size = config["mechanism_models"]["image"]["params"]["batch_size_val"]

    scm = SCM(checkpoint_dir=config["checkpoint_dir"],
              graph_structure=config["causal_graph"],
              temperature=args.sampling_temperature,
              **models)

    dataset = config["dataset"]
    data_class, unnormalize_fn = dataclass_mapping[dataset]

    transform = ReturnDictTransform(attribute_size)

    train_set = data_class(attribute_size, split='train', transform=transform)
    test_set = data_class(attribute_size, split='test', transform=transform)
    _, eval_indices, manifest = _load_benchmark_eval(test_set)

    if args.qualitative > 0:
        produce_qualitative_samples(dataset=test_set, scm=scm, parents=list(attribute_size.keys()),
                                    intervention_source=train_set, unnormalize_fn=unnormalize_fn, num=args.qualitative,
                                    show_difference=args.show_difference)

    embedding_model = get_embedding_model(args.embeddings, pretrained_vgg=True, classifier_config=args.classifier_config)
    embedding_fn = get_embedding_fn(args.embeddings, unnormalize_fn, embedding_model)

    if "effectiveness" in args.metrics:
        if dataset == "morphomnist":
            predictors = {atr: Classifier(attr=atr, num_outputs=attribute_size[atr], context_dim=len(list(config_cls["anticausal_graph"][atr])))
                          for atr in attribute_size.keys()}
        elif dataset == "celeba":
            if sum(attribute_size.values()) == 4:
                predictors = {atr: CelebaComplexClassifier(attr=atr, context_dim=len(list(config_cls["anticausal_graph"][atr])),
                                                  num_outputs=config_cls[atr +"_num_out"],
                                                  lr=config_cls["lr"], version=config_cls["version"]) for atr in attribute_size.keys()}
            else:
                predictors = {atr: CelebaClassifier(attr=atr, num_outputs=config_cls[atr +"_num_out"], lr=config_cls["lr"]) for atr in attribute_size.keys()}
        else:
            attribute_ids = get_attribute_ids(attribute_size)
            predictors = {atr: ADNIClassifier(attr=atr, num_outputs=config_cls["attribute_size"][atr], children=config_cls["anticausal_graph"][atr],
                                              num_slices=config_cls["attribute_size"]['slice'], attribute_ids=attribute_ids, arch=config_cls['arch']) for atr in attribute_size.keys()}

        # load checkpoints of the predictors
        for key , cls in predictors.items():
            print(key)
            file_name = next((file for file in os.listdir(config_cls["ckpt_path"]) if file.startswith(key)), None)
            print(file_name)
            cls.load_state_dict(torch.load(config_cls["ckpt_path"] + file_name , map_location=torch.device('cuda'))["state_dict"])
            cls.to('cuda')
            cls.eval()

        for pa in attribute_size.keys():
            evaluate_effectiveness(
                test_set, unnormalize_fn, batch_size, scm=scm,
                attributes=list(attribute_size.keys()), do_parent=pa,
                intervention_source=train_set, predictors=predictors, dataset=dataset,
                eval_indices=eval_indices, manifest=manifest,
            )

    if "fid" in args.metrics:
        evaluate_fid(
            real_set=train_set, test_set=test_set, batch_size=batch_size, scm=scm,
            attributes=list(attribute_size.keys()),
            eval_indices=eval_indices, manifest=manifest,
        )

    if "minimality" in args.metrics:
        evaluate_minimality(
            real_set=train_set, test_set=test_set, batch_size=batch_size, scm=scm,
            attributes=list(attribute_size.keys()),
            embedding=args.embeddings, embedding_fn=embedding_fn,
            eval_indices=eval_indices, manifest=manifest,
        )
