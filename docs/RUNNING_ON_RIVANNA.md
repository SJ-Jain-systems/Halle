# Running this pipeline on Rivanna (UVA HPC)

This walks through every stage from a bare Rivanna account to the final
year-by-year / by-stage demographic representativeness summary. It assumes
familiarity with `sbatch`/`squeue` but not with this specific repo.

**Caveat up front:** this repo was built in a sandbox with no access to
Rivanna, no GPU, and no network path to `huggingface.co` or PLOS's
infrastructure, so most of this hasn't been run end-to-end on real Rivanna
hardware. Module names, partition names, and GPU types are Rivanna's as of
early 2026 — run `module avail`, `allocations`, and `sinfo -o "%P %G"` to
confirm current names before submitting, since these do change. Everything
that *can* be verified without a GPU (the parsing, filtering, sampling, and
aggregation logic, plus step 1's `PLOS_CORPUS` env var and download function
— confirmed against the actual installed `allofplos` package source, not
guessed) has 25 passing unit tests — see `pytest` output — so the stages you
should trust immediately are 1–3 below; treat the GPU stages as "designed
and ready to debug," not "guaranteed to run unmodified."

## Pipeline stages

```
allofplos corpus (local XML mirror)
        │  src/build_corpus_index.py
        ▼
data/corpus_index.csv           ← full filtered population (brief item 3)
        │  src/sample_articles.py
        ▼
data/sampled_articles.csv       ← 10-article pilot
        │  src/run_pilot.py     (Llama-3.3-70B-Instruct, docs/DECISIONS.md #3)
        ▼
results/<model>/<doi>.json      ← QA spot-check against docs/SCORING_RUBRIC.md
        │
        │  src/run_pipeline.py (chosen model, full corpus_index.csv)
        ▼
data/demographics_table.shard*.csv
        │  src/merge_shards.py
        ▼
data/demographics_table.csv     ← one row per demographic sample, all articles
        │  src/analyze_trends.py
        ▼
data/analysis/*.csv, *.png      ← answers the brief's main goal
```

## Step 0: environment setup

Run this on a login node (compute nodes typically have no outbound
internet, so anything that downloads — packages, corpus, model weights —
needs to happen here or in a job with explicit internet access).

```bash
git clone <this-repo-url> ~/halle && cd ~/halle

module purge
module load miniforge   # `module avail miniforge anaconda python` if this name is wrong
conda create -n halle python=3.11 -y
conda activate halle

pip install -r requirements.txt
```

The `torch`/`vllm` lines in `requirements.txt` are CUDA-version-specific.
Check which CUDA module Rivanna's GPU nodes expect (`module avail cuda`)
and install a matching torch build if the plain `pip install torch` doesn't
pick the right one, e.g.:

```bash
module load cuda/12.4.1   # match whatever `module avail cuda` shows
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install vllm
```

Verify the CPU-only parts of the pipeline work before touching a GPU:

```bash
pytest   # should show 25 passed
```

## Step 1: get the allofplos corpus onto Rivanna

This is the data-access decision in `docs/DECISIONS.md` #1 — once this step
is done, nothing else in the pipeline touches the network.

1. Pick a storage location with real quota — **not your home directory**.
   `/scratch/$USER` or your group's `/project` allocation are the usual
   choices on Rivanna; check available space first (`df -h /scratch/$USER`).
   The corpus zip covers **every PLOS journal**, not just PLOS ONE — allofplos
   doesn't filter at download time (`MIN_FILES_FOR_VALID_CORPUS = 200000` in
   its source, so expect a large, multi-hour download+extract). Our own
   `src/build_corpus_index.py` does the PLOS ONE / subfield filtering
   afterward, locally.
2. Set the corpus directory (allofplos's own env var — confirmed against the
   installed package source, `allofplos.get_corpus_dir()`):
   ```bash
   export PLOS_CORPUS=/scratch/$USER/allofplos_corpus
   mkdir -p "$PLOS_CORPUS"
   ```
   Add that `export` to your shell profile (or the top of each SLURM script)
   so every later step finds it — `src/allofplos_client.py` reads this same
   variable.
3. Run the sync. `pip install allofplos` has no CLI entrypoint (no console
   script) — call the download function directly. This is a large, long-running
   download; run it inside `tmux`/`screen` on the login node so it survives a
   disconnected session:
   ```bash
   tmux new -s corpus-sync
   python -c "from allofplos.corpus.plos_corpus import create_local_plos_corpus; create_local_plos_corpus(directory='$PLOS_CORPUS')"
   # Ctrl-b d to detach; `tmux attach -t corpus-sync` to check back in
   ```
   This downloads `https://allof.plos.org/allofplos.zip` and extracts every
   article XML into `$PLOS_CORPUS`. You'll see `tqdm` progress bars for both
   the download and the extraction.
4. Sanity check:
   ```bash
   python -c "from src.allofplos_client import corpus_size; print(corpus_size())"
   ```
   should print a large number (hundreds of thousands) once the sync is done.

## Step 2: build the filtered corpus index

```bash
sbatch slurm/build_index.slurm
```

CPU-only, no GPU needed. Writes `data/corpus_index.csv` — every PLOS ONE
research article tagged with one of the five target psychology subfields,
2010–2026 (brief item 3). Check `wc -l data/corpus_index.csv` and spot-check
a few rows' `subject`/`subject_level_1`/`matched_subfields` columns against
the actual articles on plos.org — the taxonomy-parsing logic in
`src/jats_xml.py` is unit-tested against a synthetic fixture, not against
real PLOS XML, so this is the first point where it's worth a manual look.

## Step 3: draw the 10-article pilot sample

```bash
python -m src.sample_articles --index data/corpus_index.csv --out data/sampled_articles.csv
```

Fast, CPU-only — fine to run directly on the login node. Writes
`data/sampled_articles.csv`, 2 articles from each of the 5 subfields
(`docs/DECISIONS.md` #2). Re-run with `--seed <n>` if you want a different
draw; the default seed (42) is reproducible.

## Step 4: pilot run — QA the chosen model before scaling up

`meta-llama/Llama-3.3-70B-Instruct` is gated on Hugging Face — accept Meta's
license on the model page, then on the login node:

```bash
huggingface-cli login   # paste a token with read access, after accepting
                         # the Llama-3.3 license at huggingface.co
export HF_HOME=/scratch/$USER/hf_cache   # keep weights off your home quota
```

Submit the pilot job:

```bash
sbatch slurm/run_pilot.slurm
```

Writes `results/meta-llama__Llama-3.3-70B-Instruct/<doi>.json` for all 10
pilot articles. This isn't a model comparison — the model choice is settled
(`docs/DECISIONS.md` #3) — it's a QA spot-check: score the output against
`docs/SCORING_RUBRIC.md` (coverage, numeric accuracy, schema adherence,
multi-sample handling) to catch a bad prompt or a parsing bug on 10 articles
rather than after burning GPU hours on the full corpus.

**If Llama-3.3-70B-Instruct doesn't fit your GPU allocation** (needs
roughly 140GB+ of GPU memory in bf16 across the tensor-parallel group): drop
to an AWQ/GPTQ quantized checkpoint of the same model, or reduce
`--tensor-parallel-size` only if you have enough total GPU memory some
other way — don't reduce it below what the model needs to fit, vLLM will
just OOM.

## Step 5: run the full pipeline

Once the pilot output looks right, run every article in
`data/corpus_index.csv` through the same model:

```bash
sbatch slurm/run_pipeline.slurm
```

This is a SLURM **array job** — each array task processes a shard of the
index independently and writes its own `data/demographics_table.shard<N>.csv`
(see the script's header comment for why: avoids multiple tasks writing to
one CSV concurrently, and makes preemption/retry safe since
`src/run_pipeline.py` skips DOIs already present in its shard's output).
Adjust `SHARD_COUNT` and `--array=0-N` together, and the `--time`/`--gres`
requests, based on how many articles ended up in `corpus_index.csv` and how
much GPU capacity your allocation actually has.

Monitor with `squeue -u $USER`; check `logs/run_pipeline_*.out` for
per-shard progress (`src/run_pipeline.py` logs every 50 articles).

When all array tasks finish:

```bash
python -m src.merge_shards --glob "data/demographics_table.shard*.csv" --out data/demographics_table.csv
```

(`slurm/analyze_trends.slurm` does this merge automatically before running
the next step, if you'd rather submit one job for both.)

## Step 6: analyze trends — this answers the research question

```bash
sbatch slurm/analyze_trends.slurm
# or, if you already merged shards and just want to iterate on the analysis:
python -m src.analyze_trends --table data/demographics_table.csv --out-dir data/analysis
```

Writes, to `data/analysis/`, for each of gender / race / education:
- `<category>_reporting_rate_by_year.csv` and `..._by_stage.csv` — % of
  samples reporting that category
- `<category>_mean_pct_by_year.csv` and `..._by_stage.csv` — mean reported
  percentage per subgroup (e.g. mean % female), among samples that reported it
- `<category>_reporting_rate_by_year.png` — trend-line plot of the above

Plus `ses_reporting_detail_by_year.csv`/`..._by_stage.csv` (socioeconomic
status uses the brief's 0/1/2 reporting-detail scale rather than a simple
reported/not split).

These CSVs are the direct answer to the brief's main goal: representativeness
of demographic reporting, year by year and by stage
(2010–2014 / 2015–2019 / 2020–2023 / 2024–2026).

## Troubleshooting

- **`ModuleNotFoundError` for a package you know you installed (e.g. `lxml`)
  when a SLURM script runs, even though `pytest` works fine interactively**:
  `conda activate halle` can silently fail to rewrite `PATH` in some Rivanna
  shell configurations — `conda env list` will still claim `halle` is
  active, but `python`/`sys.executable` resolve to the base miniforge
  install instead. Confirmed live, not hypothetical. The SLURM scripts in
  this repo work around it by calling
  `$HOME/.conda/envs/halle/bin/python` directly instead of relying on
  `conda activate` — if you hit this outside those scripts, use the same
  absolute-path workaround rather than debugging `conda activate` further.
- **`CorpusNotFoundError` from `src/allofplos_client.py`**: `PLOS_CORPUS`
  isn't set or doesn't contain XML files yet — redo step 1.
- **vLLM OOM on model load**: the tensor-parallel size doesn't provide
  enough combined GPU memory for the model; request more/bigger GPUs or
  switch to a quantized checkpoint.
- **`run_pilot`/`run_pipeline` can't reach huggingface.co from a compute
  node**: pre-download weights on the login node first (`huggingface-cli
  download <model_id>` into `$HF_HOME`) so the compute node loads from local
  cache instead of fetching live.
- **`src/build_corpus_index.py` keeps almost nothing**: check that
  `TARGET_JOURNAL_SUBSTRING`/`TARGET_ARTICLE_TYPE` in
  `src/build_corpus_index.py` actually match the strings in your corpus's
  XML — these were written from the standard JATS layout PLOS has
  documented, but confirm against a real file with
  `python -c "from src.jats_xml import parse_metadata; print(parse_metadata('<path-to-a-real-xml-file>'))"`.
