# CounterStroke

This is the code for the paper *CounterStroke: Investigating Evolutionary Image Counterfactual Generation*.

CounterStroke looks for a visual counterfactual by evolving a drawing program and painting it onto the original image. The classifier is used as a black box: the search only needs class probabilities. It does not need gradients, internal activations, or a generative model trained to produce counterfactuals.

A candidate is an integer genome. A grammar maps that genome to a sketch (the phenotype): one or more turtle paths, each with a start point, a short sequence of steps, a stroke or a fill, and a signed change in intensity. On MNIST and CIFAR-10 the same mask is applied to every colour channel. On ImageNet-100 each shape picks one of red, green, or blue. The turtle is kept inside the image, so a step that would leave the grid is shortened.

Each image gets its own population. NSGA-II minimises three objectives:

1. Decision margin. For the original class `y`, this is `p(y)` minus the highest probability of any other class. A negative margin means the prediction has flipped.
2. Sparsity. How many pixel values changed, with a tolerance of `1e-6`.
3. Proximity. Euclidean distance from the original image.

If the margin is still non-negative, sparsity and proximity are replaced by a large penalty, so an almost unchanged image cannot win. From the valid members of the final population the code keeps three picks, which are the three CounterStroke columns in the paper:

- sparsity first, then proximity (`cf_graco_base_o2`, called CS-sparse in the paper)
- proximity first, then sparsity (`cf_graco_base_o3`, CS-prox)
- the same sparsity-first rule, but only among candidates whose top-class probability is at least 0.5 (`cf_graco50`, CS-50)

Those three are different readouts of one search, not three separate trainings.

## Classification experiments

The main comparison is on three classifiers, with 100 fixed evaluation images:

| Dataset | Images | Classifier | Eval set |
| --- | --- | --- | --- |
| MNIST | 28×28 grey | two-layer CNN | 100 test images, 10 per class |
| CIFAR-10 | 32×32 RGB | ResNet-18 | 100 test images, 10 per class |
| ImageNet-100 | 299×299 RGB | fine-tuned Inception-v3 | 100 validation images, 1 per class |


The baselines in the pipeline are:

- gradient counterfactual (Wachter-style input optimisation)
- CFProto (nearest latent prototype of the target class)
- CFRL
- LeapFactual (latent conditional flow)

Per-image metrics include whether the prediction flipped, how many pixels changed, L2 distance, and perceptual distances. Set-level scores include FID, KID, and CMMD.

## Setup

Run everything from this directory, on a machine with a GPU. The datasets and the trained weights are not in the repository.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running a classification pipeline

MNIST is downloaded into `MNIST/data` on the first run.

```bash
python MNIST/run_pipeline.py --mnist-root MNIST/data
```

CIFAR-10 is cached by torchvision under `Cifar10/data`.

```bash
python Cifar10/run_pipeline.py --cifar-root Cifar10/data
```

ImageNet-100 must already be on disk, in ImageFolder layout (`train/` and `val/`).

```bash
python Imagenet/run_pipeline.py --imagenet-root /path/to/imagenet100
```

Each pipeline trains the classifier, builds the confusion matrix, selects the 100 evaluation images, trains the baseline explainers on the training split, generates counterfactuals, runs CounterStroke, and writes metrics and plots. A phase that has already finished is skipped. Pass `--only <phase>` to run one phase, `--force-phase <phase>` to repeat one, or `--skip-graco` to skip the evolutionary phase.

Phase names differ slightly by dataset. On MNIST they are `train_classifier`, `train_explainer_models`, `compute_confusion_csv`, `select_eval_samples`, `generate_counterfactual_pt`, `generate_cfproto`, `generate_cfrl`, `generate_leapfactual`, `generate_graco_v8`, and `metrics_and_plots`. CIFAR-10 uses `generate_graco_cifar`. ImageNet-100 uses `generate_graco_inception` and also has `download_imagenet100`.

Counterfactual tensors for one image land in `tensors.npz` under that dataset's generation folder. The three CounterStroke arrays are `cf_graco_base_o2`, `cf_graco_base_o3`, and `cf_graco50`.

## Deep-SCM interventions

On MorphoMNIST and CelebA the question is not "did the class flip?" but "does the image express a requested intervention?" The first objective becomes an intervention loss from the benchmark's own attribute models. CounterStroke does not train those attribute models, and it does not use the VAE, HVAE, or GAN to draw the image. Sparsity and proximity stay as the other two objectives. If no candidate meets the intervention, the pick falls back to the smallest intervention loss.

MorphoMNIST covers digit identity, thickness, and intensity (200 test indices, 600 interventions). CelebA is run on the benchmark's simple graph (eyeglasses, smiling; 400 interventions) and complex graph (bald, male, no-beard, young; 800 interventions). Images and the SCM checkpoints are downloaded or trained separately.

```bash
python benchmark/morphomnist/run_pipeline.py
python benchmark/celeba/run_pipeline.py
```

## Other analyses in the paper

Repeated evolution runs, for the mean-best curves of the three objectives and the pooled Pareto fronts:

```bash
python evolution_experiment/run_experiment.py
```

A smaller evaluation budget. `--dataset` also accepts `cifar` and `imagenet`. For ImageNet pass `--imagenet-root`.

```bash
python eval_budget/run_budget.py --dataset mnist --stage all
```

The grammar on its own, with selection, crossover, and mutation turned off. This is the random-ablation entry point. It runs MNIST, then CIFAR-10, then ImageNet-100, and writes one JSON. Stopping and running the same command again continues. ImageNet-100 needs `--imagenet-root`.

```bash
python random_ablation/run_random_ablation.py --device cuda --imagenet-root /path/to/imagenet100
```

Blank canvas. The same grammar is evolved from a uniform image toward each class, with no factual photo. MNIST starts from black, CIFAR-10 and ImageNet-100 from white. Sparsity and proximity are measured against that canvas. Each sketch is later paired with the real validation image of the target class that has the highest SSIM.

```bash
python analysis/mnist/run_blank_graco_pdf.py
python analysis/cifar10/run_blank_graco_pdf.py
python analysis/imagenet100/run_blank_graco_pdf.py --data-root /path/to/imagenet100
```

Phenotype attribution. Successful high-confidence flips are aggregated into an importance map (sum and mean intensity, change frequency, a strict co-change rule, a smoothed frequency map, and PageRank on co-edited pixels). Those maps are scored with symmetric relevance gain against integrated gradients, input times gradient, SmoothGrad, occlusion, Grad-CAM, and SHAP.

```bash
python explainability/scripts/run_batch.py
```

The Streamlit app browses the final populations and the precomputed metrics. The PDF script lays the original image next to every method for a sample.

```bash
./analyse_final_population/run_app.sh
python visualization/make_comparison_pdfs.py
```
