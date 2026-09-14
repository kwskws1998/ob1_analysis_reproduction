# Standalone OB1 attention-analysis reproduction

This directory reproduces the Provo/OB1 analysis without importing code or
outputs from its parent repository. It downloads the external assets at pinned
revisions, runs ET1 on all 55 Provo passages, independently simulates 100 OB1
readers under each of `attention_skew=3` and `attention_skew=4`, analyzes six
frozen Llama-3-8B kernel parameter pairs, and generates the final table and two
figures.

## Run from a fresh copy

Python 3.12 is required. Every executable is a Python file; no shell script is
used.

```bash
cd ob1_analysis_reproduction

python3.12 setup_environment.py

.venv/bin/python run_all.py --workers 40

.venv/bin/python verify_results.py
```

The default worker count is `min(40, CPU count - 4)`. `--workers` changes only
OB1 process concurrency, not the requested seeds or analysis settings.

Before a parallel run, one short cache-only process builds and validates the
OB1 frequency, lexicon, prediction, and inhibition files. This process enters
no passage-reading loop and simulates zero readers. Immediately after cache
validation, all seeds 0 through 99 are distributed across the requested worker
count; with `--workers 40`, forty simulation processes therefore start together.

Individual stages can be rerun as follows:

```bash
.venv/bin/python download_assets.py
.venv/bin/python run_et1_prediction.py
.venv/bin/python run_ob1_simulations.py --workers 40
.venv/bin/python analyze_attention.py
.venv/bin/python make_figures.py
.venv/bin/python verify_results.py
```

After assets have been downloaded, optional one-passage ET1 and OB1 runtime
smoke tests can be enabled with:

```bash
OB1_RUN_INTEGRATION_TESTS=1 .venv/bin/python -m unittest discover -s tests -v
```

A nonempty stage directory is reused only after its full manifest and output
contract pass validation. An incomplete or incompatible directory causes an
error instead of being mixed with a new run.

## Exact analysis contract

`run_et1_prediction.py` writes both `et1_predictions.csv` and
`t5_token_grid.csv`. The first contains the actual ET1 token-level TRT
predictions. The OB1 comparison reads only the second file's T5 token indices
and character spans. It fixes the source TRT to one, so ET1-predicted TRT
magnitudes do not enter Spearman correlation or Jensen-Shannon divergence.

For each skew, OB1 trajectories are generated from Provo passages with seeds
0 through 99. The skew-3 and skew-4 analyses are rejected unless the trajectory
manifest's generation skew equals the attention-formula skew used in that
analysis. Both conditions must contain all 55 passages, and their fixation
files must differ.

At every saved fixation, the analysis evaluates OB1's focused Gaussian
attention in letter coordinates and maps it to relative native T5-token
positions. All candidates are normalized on the mapped tokens visible at that
fixation and pooled with fixation-duration weights. The three reported
conditions are:

- no redistribution: a unit impulse at the source token;
- paired symmetric control: equal left/right Gaussian scales selected to match
  the learned kernel's centered token-position standard deviation in the same
  fixation contexts;
- learned asymmetric redistribution: each checkpoint's frozen left and right
  sigma values.

Metrics are computed separately for each checkpoint and then averaged with
equal checkpoint weight. Spearman correlation uses offsets with mass in either
distribution. Jensen-Shannon divergence uses the entire normalized offset
distribution and is SciPy's base-2 Jensen-Shannon distance squared.

## Provenance boundaries

The six sigma pairs and downstream accuracies are stored in
`config/llama3_8b_six_checkpoints.json`; full reward-model checkpoints are not
required. None of those sigma values is fitted to Provo or OB1.

The GazeReward paper describes ET1 word for word as follows:

> “This model was trained on the Dundee, GECO, ZuCo1, and ZuCo2 datasets, and predicts total reading time (TRT) per token.”

Source: [GazeReward, p. 7](https://arxiv.org/pdf/2410.01532).

The OB1 paper defines its directional attention parameter word for word:

> “Asym is equal to 1 toward the right and 0.25 toward the left”

Source: [Snell et al. (2018), p. 973](https://research.vu.nl/ws/portalfiles/portal/72578613/OB1_reader_A_model_of_word_recognition_and_eye_movements_in_text_reading.pdf).

The complete URLs, commits, sizes, and SHA-256 values are frozen in
`asset_manifest.json`. The OB1 snapshot and SUBTLEX-UK are downloaded at run
time because no license file or explicit redistribution grant was observed in
the pinned sources. The downloader verifies archives before extraction and
rejects archive traversal and links.

## Outputs and verification

The final directory is `outputs/final/` and contains:

- `reviewer_metric_table.csv` and `.md`;
- `mean_metric_table.csv`;
- `mean_offset_profiles.csv`;
- `mean_region_mass.csv`;
- `ob1_mean_profiles.png` and `.pdf`;
- `ob1_region_mass.png` and `.pdf`;
- `figure_audit.json`;
- `provenance_manifest.json`;
- `execution.log` when produced through `run_all.py`.

`verify_results.py` checks the pinned inputs, 55/2,686/59 Provo counts, 3,715
ET1 tokens, both 100-reader trajectory grids, skew-specific fixation row
counts, six checkpoint analyses, different skew-3/skew-4 trajectory hashes,
figure-source tables, and the final reviewer metrics at absolute tolerance
`1e-6`. Figure byte hashes are intentionally not fixed because font rendering
can vary by operating system.

The comparison is a kernel-level consistency analysis of OB1's fixation-onset
spatial allocation component. It is not direct validation against human gaze
and does not claim to recover complete OB1 reading behavior.
