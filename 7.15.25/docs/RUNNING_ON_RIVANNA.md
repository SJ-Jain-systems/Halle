# Running this pipeline on Rivanna (UVA HPC)

This covers every stage, from a bare Rivanna account to the final year by year and
by-stage demographic representativeness summary. It assumes you're comfortable
with `sbatch` and `squeue`, but not that you already know this repo.

**One caveat up front:** this repo was built in a sandbox with no access to
Rivanna, no GPU, and no network path to `huggingface.co` or PLOS's infrastructure,
so most of it hasn't been run end to end on real Rivanna hardware. The module
names, partition names, and GPU types are Rivanna's as of early 2026, so run
`module avail`, `allocations`, and `sinfo -o "%P %G"` to confirm the current names
before you submit, since these change. Everything that can be checked without a
GPU (the parsing, filtering, sampling, and aggregation logic, plus step 1's
`PLOS_CORPUS` env var and download function, which we confirmed against the actual
installed `allofplos` package source rather than guessing) has 25 passing unit
tests (see the `pytest` output). So stages 1 to 3 below are the ones to trust
immediately; treat the GPU stages as designed and ready to debug, not guaranteed
to run untouched.

## Pipeline stages

```
allofplos corpus (local XML mirror)
        │  src/build_corpus_index.py
        ▼
data/corpus_index.csv           ← full filtered population (brief item 3)
        │  src/sample_articles.py
        ▼
data/sampled_articles.csv       ← 46-article pilot
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

Run this on a login node (compute nodes usually have no outbound internet, so
anything that downloads, whether packages, the corpus, or model weights, has to
happen here or in a job with explicit internet access).

```bash
git clone <this-repo-url> ~/halle && cd ~/halle

module purge
module load miniforge   # `module avail miniforge anaconda python` if this name is wrong
conda create -n halle python=3.11 -y
conda activate halle

pip install -r requirements.txt
```

The `torch` and `vllm` lines in `requirements.txt` depend on your CUDA version.
Check which CUDA module Rivanna's GPU nodes expect (`module avail cuda`) and
install a matching torch build if a plain `pip install torch` doesn't select the
right one, for example:

```bash
module load cuda/12.4.1   # match whatever `module avail cuda` shows
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install vllm
```

Confirm the CPU-only parts of the pipeline work before you touch a GPU:

```bash
pytest   # should show 25 passed
```

## Step 1: get the allofplos corpus onto Rivanna

This is the data-access decision in `docs/DECISIONS.md` #1. Once this step is done,
nothing else in the pipeline touches the network.

1. Pick a storage location with real quota, **not your home directory**.
   `/scratch/$USER` or your group's `/project` allocation are the usual choices on
   Rivanna; check the free space first (`df -h /scratch/$USER`). The corpus zip
   covers **every PLOS journal**, not just PLOS ONE, since allofplos doesn't filter
   at download time (`MIN_FILES_FOR_VALID_CORPUS = 200000` in its source, so expect
   a large download and extract that takes a few hours). Our own
   `src/build_corpus_index.py` does the PLOS ONE and subfield filtering afterward,
   locally.
2. Set the corpus directory (allofplos's own env var, confirmed against the
   installed package source, `allofplos.get_corpus_dir()`):
   ```bash
   export PLOS_CORPUS=/scratch/$USER/allofplos_corpus
   mkdir -p "$PLOS_CORPUS"
   ```
   Add that `export` to your shell profile (or the top of each SLURM script) so
   every later step finds it. `src/allofplos_client.py` reads the same variable.
3. Run the sync. `pip install allofplos` has no CLI entrypoint (no console script),
   so call the download function directly. This is a large, long-running download,
   so run it inside `tmux` or `screen` on the login node so it survives a dropped
   session:
   ```bash
   tmux new -s corpus-sync
   python -c "from allofplos.corpus.plos_corpus import create_local_plos_corpus; create_local_plos_corpus(directory='$PLOS_CORPUS')"
   # Ctrl-b d to detach; `tmux attach -t corpus-sync` to check back in
   ```
   This downloads `https://allof.plos.org/allofplos.zip` and extracts every article
   XML into `$PLOS_CORPUS`. You'll see `tqdm` progress bars for the download and the
   extraction.
4. Sanity check:
   ```bash
   python -c "from src.allofplos_client import corpus_size; print(corpus_size())"
   ```
   should print a large number (hundreds of thousands) once the sync is done.

## Step 2: build the filtered corpus index (from PLOS Solr)

```bash
python -m src.build_index_solr --out data/corpus_index.csv --corpus-dir "$PLOS_CORPUS"
```

**Run this on the login node** (it needs internet, and compute nodes don't have
it). It's limited by the network, not the GPU: it queries every psychology PLOS
ONE research article from 2010 to 2026 directly from PLOS's Solr index and maps
each DOI to its local XML file. Takes roughly 10 to 20 minutes.

Why Solr instead of scanning the local XML: the allofplos XML is missing the
subject taxonomy for a large share of some years (about 99% of 2015, about 37% of
2013, see `scripts/diagnose_2015.py`), so an XML-only scan drops those articles.
PLOS's Solr index has the authoritative taxonomy for every article. This is the
"Solr for discovery, allofplos for full text" split from the brief. (The old
XML-scan path, `slurm/build_index.slurm` and `src/build_corpus_index.py`, is kept
in the repo for reference but this step supersedes it.)

Sanity-check when it finishes. The per-year counts should be smooth, with no
near-empty years (the bug this step fixes):

```bash
python -c "
import csv
from collections import Counter
rows=list(csv.DictReader(open('data/corpus_index.csv')))
print('total:', len(rows))
print('by year:', dict(sorted(Counter(r['year'] for r in rows).items())))
"
```

## Step 3: draw the pilot sample

```bash
python -m src.sample_articles --index data/corpus_index.csv --out data/sampled_articles.csv
```

Fast and CPU-only, so it's fine to run on the login node. Writes
`data/sampled_articles.csv`, 2 articles per psychology subfield in the index
(`docs/DECISIONS.md` #2, so 2×N articles for N subfields, not a fixed 10, since
PLOS uses more than the brief's five). Re-run with `--seed <n>` for a different
draw; the default seed (42) is reproducible.

## Step 4: pilot run, QA the model before scaling up

`meta-llama/Llama-3.3-70B-Instruct` is gated on Hugging Face, so accept Meta's
license on the model page, then on the login node:

```bash
huggingface-cli login   # paste a token with read access, after accepting
                         # the Llama-3.3 license at huggingface.co
export HF_HOME=/scratch/$USER/hf_cache   # keep weights off your home quota
```

First, dry-run the wiring on the login node, with no GPU and no model download:

```bash
python -m src.run_pilot --sample data/sampled_articles.csv --backend echo
```

The `echo` backend loads no model. It runs the whole path (read the sample CSV,
find each article's local XML, build the prompt, validate the schema, write
`results/.../<doi>.json`) and surfaces any path or parsing problem before you spend
a GPU allocation. The output values are placeholders, not real extractions.

Then submit the real pilot job:

```bash
sbatch slurm/run_pilot.slurm
```

Writes `results/meta-llama__Llama-3.3-70B-Instruct/<doi>.json` for all 46 pilot
articles. The run is resumable: an article already written with `status: "ok"` is
skipped, so a preempted job resumes where it left off (pass `--overwrite` to force
a full re-run). This is not a model comparison; the model choice is settled
(`docs/DECISIONS.md` #3). It's a QA spot-check: score the output against
`docs/SCORING_RUBRIC.md` (coverage, numeric accuracy, schema adherence,
multi-sample handling) to catch a bad prompt or a parsing bug on 46 articles
rather than after spending GPU hours on the full corpus.

**If Llama-3.3-70B-Instruct doesn't fit your GPU allocation** (it needs roughly
140GB or more of GPU memory in bf16 across the tensor-parallel group): drop to an
AWQ or GPTQ quantized checkpoint of the same model, or lower
`--tensor-parallel-size`, but only if you have enough total GPU memory another way.
Don't drop it below what the model needs to fit, or vLLM will OOM.

## Step 5: run the full pipeline

Once the pilot output looks right, run every article in `data/corpus_index.csv`
through the same model:

```bash
sbatch slurm/run_pipeline.slurm
```

This is a SLURM **array job**: each array task processes one shard of the index
independently and writes its own `data/demographics_table.shard<N>.csv` (the
script's header comment explains why: it avoids several tasks writing to one CSV at
once, and it makes preemption and retry safe, since `src/run_pipeline.py` skips
DOIs already in its shard's output). Adjust `SHARD_COUNT` and `--array=0-N`
together, plus the `--time` and `--gres` requests, based on how many articles ended
up in `corpus_index.csv` and how much GPU capacity your allocation has.

Monitor it with `squeue -u $USER`; check `logs/run_pipeline_*.out` for per-shard
progress (`src/run_pipeline.py` logs every 50 articles).

When all the array tasks finish:

```bash
python -m src.merge_shards --glob "data/demographics_table.shard*.csv" --out data/demographics_table.csv
```

(`slurm/analyze_trends.slurm` runs this merge for you before the next step, if
you'd rather submit one job for both.)

## Step 6: analyze trends, the step that answers the question

```bash
sbatch slurm/analyze_trends.slurm
# or, if you already merged shards and just want to iterate on the analysis:
python -m src.analyze_trends --table data/demographics_table.csv --out-dir data/analysis
```

Writes, into `data/analysis/`, for each of gender / race / education:
- `<category>_reporting_rate_by_year.csv` and `..._by_stage.csv`: the % of samples
  reporting that category
- `<category>_mean_pct_by_year.csv` and `..._by_stage.csv`: the mean reported
  percentage per subgroup (for example mean % female), among samples that reported
  it
- `<category>_reporting_rate_by_year.png`: a trend-line plot of the above

Plus `ses_reporting_detail_by_year.csv` and `..._by_stage.csv` (socioeconomic
status uses the brief's 0/1/2 reporting-detail scale rather than a simple
reported-or-not split).

These CSVs are the direct answer to the brief's main goal: how representative
demographic reporting is, year by year and by stage (2010 to 2014, 2015 to 2019,
2020 to 2023, 2024 to 2026).

## Troubleshooting

- **`ModuleNotFoundError` for a package you know you installed (for example `lxml`)
  when a SLURM script runs, even though `pytest` works fine interactively:**
  `conda activate halle` can quietly fail to fix up `PATH` in some Rivanna shell
  setups. `conda env list` will still say `halle` is active, but `python` and
  `sys.executable` point at the base miniforge install instead. We saw this in
  practice; it's not hypothetical. The SLURM scripts in this repo avoid it by
  calling `$HOME/.conda/envs/halle/bin/python` directly instead of relying on
  `conda activate`. If you hit it outside those scripts, use the same absolute-path
  workaround rather than debugging `conda activate` further.
- **`CorpusNotFoundError` from `src/allofplos_client.py`:** `PLOS_CORPUS` isn't set,
  or it doesn't have XML files in it yet. Redo step 1.
- **vLLM OOM on model load:** the tensor-parallel size isn't giving the model
  enough combined GPU memory. Request more or bigger GPUs, or switch to a quantized
  checkpoint.
- **`run_pilot` or `run_pipeline` can't reach huggingface.co from a compute node:**
  pre-download the weights on the login node first (`huggingface-cli download
  <model_id>` into `$HF_HOME`) so the compute node loads from the local cache
  instead of fetching live.
- **`src/build_corpus_index.py` keeps almost nothing:** check that
  `TARGET_JOURNAL_SUBSTRING` and `TARGET_ARTICLE_TYPE` in
  `src/build_corpus_index.py` actually match the strings in your corpus's XML.
  These were written from the standard JATS layout PLOS documents, so confirm
  against a real file with
  `python -c "from src.jats_xml import parse_metadata; print(parse_metadata('<path-to-a-real-xml-file>'))"`.
