# On Hyperparameter Tuning on the Test Set — code and results

Code and merged results for all experiments reported in the paper.

Each benchmark measures the **adaptive overfitting gap**

    Δ_AO = A(λ*_test, D_test) − A(λ*_val, D_test)

over `K = 30` paired runs with a random-search budget of `T = 30` per run.

## Layout

    mnist1d/        MNIST-1D, 8 MLPs
    cifar10/        CIFAR-10, 8 architectures, plus the CIFAR-10.1 transfer evaluation
    mnist1d_t100/   MLP-Giant rerun at T = 100 on larger fixed splits
    glue/           GLUE RTE / MRPC / CoLA, 8 pretrained transformers
    utils/          shared statistics and plotting code
    figures/        scripts that build the figures in the paper
    figures_out/    rendered figures (created by the scripts in figures/)

Each benchmark directory holds its own `benchmark.py` (sampler, trainer,
statistics), `models.py` (architecture registry), and the scripts below.
Merged results are committed under `results/` or `output/`, so every figure
and table can be reproduced without re-running any training.

## Reproducing the figures

Everything in the paper is built from the committed CSVs:

    cd figures
    python make_p_vs_T.py               # sensitivity vs budget T
    python make_per_run_gap.py          # per-run gap vs run-to-run noise
    python make_linear_fit.py           # val-tuned vs test-tuned, OLS fit
    python make_upper_bound.py          # memorisation upper bound
    python make_winner_distributions.py # hyperparameter range validity
    python make_robustness_t100.py      # MLP-Giant at T = 100

`make_p_vs_T.py` re-derives the win probability from the per-trial CSVs and
cross-checks it against the value in each `rq1_gap_analysis.csv`, so a
mismatch between the raw and analysed results fails loudly.

## Re-running an experiment

Each benchmark follows the same three stages.

    # 1. train — resumable, sharded by run and by trial slice
    python run_experiment.py --arch <name> \
        --run-start 0 --run-end 30 --trial-start 0 --trial-end 30

    # 2. merge the per-shard CSVs into the analysis-level tables
    python merge_results.py

    # 3. analyse: gap, win probability, OLS fit, rank correlations
    python analysis_rq1.py

`glue/` additionally takes `--task {rte,mrpc,cola}` on every script, and
needs `python download_assets.py` once from a machine with internet access
to populate the local Hugging Face cache.

The memorisation upper bound is a separate stage, trained and tuned directly
on the bootstrap test fold:

    python run_upper_bound.py --arch <name>     # (--task too, for glue/)
    python analysis_upper_bound.py

For CIFAR-10.1:

    python eval_cifar101.py
    python merge_cifar101_eval.py
    python analysis_cifar101_tuning.py

## Determinism

All randomness derives from a single master seed. NumPy's `SeedSequence`
splits it per architecture, per run, and per source of randomness within a
run (bootstrap split, hyperparameter draw, training seed). Hyperparameters
for all `T` trials are drawn up front and only the requested slice is
executed, so sharding a run across several jobs yields exactly the same
configurations as running it in one go.

## Requirements

Python 3.10+, PyTorch, NumPy, pandas, SciPy, matplotlib; `glue/` also needs
`transformers` and `datasets`. The image experiments additionally use
torchvision.
