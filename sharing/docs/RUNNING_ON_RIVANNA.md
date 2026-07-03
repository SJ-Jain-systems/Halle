# Running the pipeline on Rivanna

Every step, from a fresh account to the final trend tables. Assumes you know
`sbatch` and `squeue` but not this project.

A few lessons from doing this for real are baked in below. Read the
troubleshooting section at the bottom before you start, it will save you time.

## The stages

```
allofplos corpus (downloaded once)
  -> build_index_solr.py   -> data/corpus_index.csv        (master list)
  -> sample_articles.py    -> data/sampled_articles.csv    (pilot batch)
  -> run_pilot.slurm       -> results/...                  (pilot, graded by hand)
  -> run_pipeline.slurm    -> data/demographics_table.*    (full run)
  -> analyze_trends.py     -> data/analysis/*              (the findings)
```

## Step 0: environment

Do this on the login node. Compute nodes have no internet, so anything that
downloads has to happen here.

```bash
git clone <repo-url> /scratch/$USER/halle && cd /scratch/$USER/halle
module load miniforge
conda create -n halle python=3.11 -y
conda activate halle
pip install -r requirements.txt
pytest        # should pass; confirms the non-GPU logic works
```

Put everything on `/scratch`, not your home directory. Home is too small for the
corpus and the model.

## Step 1: download the corpus

The article files come from allofplos. This is one big download, about 9 GB, and
it covers every PLOS journal, not just PLOS ONE. We filter down later.

```bash
export PLOS_CORPUS=/scratch/$USER/allofplos_corpus
mkdir -p "$PLOS_CORPUS"
python -c "from allofplos.corpus.plos_corpus import create_local_plos_corpus; create_local_plos_corpus(directory='$PLOS_CORPUS')"
```

`tmux` was not available on the login node when we ran this, so we just ran it in
the foreground. It took about 13 minutes. If your session might drop, submit it
as a job instead. Set `PLOS_CORPUS` again every time you log in, it does not
persist.

Check it worked:

```bash
python -c "from src.allofplos_client import corpus_size; print(corpus_size())"
```

That should print a large number, in the hundreds of thousands.

## Step 2: build the master list (from the search engine)

Run this on the login node. It needs internet. It is not heavy, about 10 to 20
minutes.

```bash
python -m src.build_index_solr --out data/corpus_index.csv --corpus-dir "$PLOS_CORPUS"
```

Why the search engine and not the files: the files are missing subject tags for
whole years (2015 especially), so scanning them drops those papers. The search
engine has the tags for every article. See DECISIONS.md.

Check the year counts are smooth with no near-empty year:

```bash
python -c "
import csv
from collections import Counter
rows=list(csv.DictReader(open('data/corpus_index.csv')))
print('total:', len(rows))
print('by year:', dict(sorted(Counter(r['year'] for r in rows).items())))
"
```

## Step 3: draw the pilot batch

Fast, runs on the login node.

```bash
python -m src.sample_articles --index data/corpus_index.csv --out data/sampled_articles.csv
```

Writes 2 papers per subfield. With 23 subfields that is 46 papers. Same batch
every time unless you change `--seed`.

## Step 4: run the pilot on a GPU (I HAVE ONLY GOTTEN THIS FAR WITHOUT TROUBLESHOOTING)

Llama-3.3-70B is gated on Hugging Face. Accept Meta's license on the model page
first, then make a read token. On the login node:

```bash
export HF_HOME=/scratch/$USER/hf_cache
hf auth login          # paste your token (the CLI is `hf`, not huggingface-cli)
```

The model weights are about 140 GB. Download them on the login node before the
job runs, because compute nodes have no internet:

```bash
hf download meta-llama/Llama-3.3-70B-Instruct
```

Then submit the pilot:

```bash
sbatch slurm/run_pilot.slurm
```

Grade the output against SCORING_RUBRIC.md. If it looks good, go on. If it
fabricates numbers or breaks format, fix the prompt in
`src/extract_demographics.py` and re-run before spending real GPU time.

If the model won't fit your GPUs (it needs roughly 140 GB of GPU memory across
the cards), use a quantized version rather than dropping the number of GPUs below
what it needs.

## Step 5: the full run

```bash
sbatch slurm/run_pipeline.slurm
```

This is a SLURM array job. It splits the master list into chunks, one per array
task, running in parallel. Each chunk writes its own file and skips papers it
already finished, so a killed task can just be resubmitted. Set the array size
and GPU request to match your allocation. Watch it with `squeue -u $USER` and the
log files.

When every chunk is done, merge them:

```bash
python -m src.merge_shards --glob "data/demographics_table.shard*.csv" --out data/demographics_table.csv
```

## Step 6: the analysis

```bash
python -m src.analyze_trends --table data/demographics_table.csv --out-dir data/analysis
```

Writes, per demographic, the reporting rate by year and by stage, the average
composition, and trend charts. These files are the answer to the study question.

## Troubleshooting

Two of these cost us real time, so they are first.

- A batch job says it can't find a package you know is installed (like `lxml`),
  even though `pytest` works fine in your terminal. Cause: `conda activate` does
  not reliably set the path inside a batch job. It looks like it worked but the
  job runs the wrong Python. Fix: call the environment's Python by full path,
  `$HOME/.conda/envs/halle/bin/python`. The SLURM scripts already do this.

- A long job started in the background of a terminal dies when your connection
  drops. The web terminal kills background processes at session end. Fix: submit
  it as a job (`sbatch`) instead of backgrounding it. That is why the heavy steps
  are jobs.

- `CorpusNotFoundError`: `PLOS_CORPUS` isn't set or is empty. Redo step 1, and
  remember to re-export `PLOS_CORPUS` after logging in.

- The model runs out of GPU memory on load: you don't have enough GPUs for it.
  Request more, or use a quantized version.

- A job can't reach huggingface.co: you're on a compute node with no internet.
  Download the weights on the login node first (step 4).
