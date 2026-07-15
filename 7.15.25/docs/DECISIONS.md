# Project decisions

Source brief: `PLOS_ONE_PROJECT.pdf`, a study of demographic representativeness in
psychology research articles published in *PLOS ONE*, 2010 to 2026.

This document records the three open decisions from the brief and the reasoning
behind each. The code that implements them is in `src/`, and how to run it on
Rivanna is in `docs/RUNNING_ON_RIVANNA.md`.

## 1. Data access: Solr for discovery, `allofplos` for full text

**Decision: get the psychology article list from PLOS's Solr index
(`src/solr_client.py`, `src/build_index_solr.py`), and use the local `allofplos`
XML corpus only for the full text.**

This is the brief's *original* plan, but we only arrived at it after trying an
allofplos-only approach and abandoning it. It's worth recording, because it's a
real data-quality finding:

1. **First attempt: allofplos only, parse the taxonomy from the local XML.** The
   plan was to sync the whole corpus once and read PLOS's subject taxonomy
   directly from each article's JATS `<subj-group>` blocks (`src/jats_xml.py`,
   `src/build_corpus_index.py`), so the pipeline would be fully offline after the
   one-time download.
2. **Why it failed:** checked against the real corpus (391k articles), the XML is
   missing the subject taxonomy for a large, year-dependent share of articles.
   About 99% of 2015 and about 37% of 2013 have only a "Research Article" heading
   and no discipline tags at all (see `scripts/diagnose_2015.py`). An XML-only
   scan therefore dropped almost all of 2015, which would gut the "middle" (2015
   to 2019) stage of the analysis. Better parsing can't fix it, because the data
   simply isn't in those files.
3. **Fix: use Solr for discovery.** PLOS's Solr index carries the authoritative
   taxonomy for every article. `src/build_index_solr.py` queries it for every PLOS
   ONE research article from 2010 to 2026 tagged under Psychology
   (`subject:"Psychology"`, with the required `doc_type:full` and strict `fq`
   filters, see `scripts/test_solr.py`), reads the subfield from the subject
   *paths*, and maps each DOI to its local XML file for the full text. Gap-free
   and consistent across all years.

The subject metadata (brief item 4.c.i: `subject` / `subject_level_1`) comes from
Solr's subject paths; the full text (Methods/Participants) still comes from the
local XML via `src/jats_xml.py::get_extraction_text()`, which is present even for
the articles with no taxonomy.

Net effect: one network step (the Solr enumeration, on the login node, about 10 to
20 minutes rather than the roughly 1.5-hour XML scan) plus the one-time corpus
download; everything after that reads local XML. The old XML-scan code
(`src/build_corpus_index.py`, `slurm/build_index.slurm`) is kept for reference but
superseded.

## 2. Pilot sample

**Decision: 2 articles per psychology subfield, across all the subfields PLOS
actually tags (not a hardcoded five), drawn at random with a fixed seed so it's
reproducible.**

The brief (item 3.c) named five subfields: social, cognitive, developmental,
clinical, and quantitative psychology. But checked against the real corpus, PLOS's
taxonomy doesn't use "Quantitative psychology" as a term at all, and it does use
several the brief didn't list (Experimental psychology, Psychometrics, and
others). Rather than force the brief's list onto a taxonomy that doesn't match it,
the pipeline is taxonomy-driven: it keeps every article PLOS files under
Psychology and records whatever subfield PLOS assigned (see decision #1 and
`jats_xml.get_psychology_subfields`). The pilot then stratifies over whatever
subfields the index actually contains.

Why stratify instead of pooling everything and sampling uniformly: a uniform draw
would let a high-volume subfield (social or cognitive) crowd out the rest, and the
point of the pilot is to test the model's extraction quality across subfields
before we spend GPU time on the full run. Two per subfield guarantees coverage.
(That's why the pilot is 2×N articles for N subfields, not a fixed 10; the brief's
"10" assumed exactly 5 subfields.)

Implementation: `src/sample_articles.py`, the `stratified_sample()` function (it
derives the subfields from the index and skips any with fewer than 2 articles),
reading from `data/corpus_index.csv` (no network). The seed defaults to `42`;
override it with `--seed`.

## 3. Model choice

**Decision: `meta-llama/Llama-3.3-70B-Instruct`. Final; this is the only model the
pipeline runs.**

The brief gave two candidates and asked us to pick one:

| | Mistral-7B-Instruct-v0.3 | Llama-3.3-70B-Instruct |
|---|---|---|
| Parameters | 7B | 70B |
| Context window | 32K | 128K |
| Structured extraction / instruction-following accuracy | Noticeably weaker on multi-field JSON extraction and on numeric reasoning buried in prose | Clearly stronger, and less likely to hallucinate percentages or drop a demographic subgroup |
| Compute/cost | Runs on a single consumer GPU or free-tier hosted inference | Needs several GPUs, a real constraint on shared infrastructure but not on a dedicated HPC allocation |

Llama-3.3-70B-Instruct wins on the criterion that matters here: a missed or
fabricated demographic percentage feeds straight into the representativeness
analysis, which is the whole point of the project, so extraction accuracy is what
we optimize for. The only argument for Mistral-7B would be compute cost, and
running on Rivanna (a dedicated GPU allocation, no per-token API cost, no rate
limits) removes that. There's no real tradeoff left, so there's no reason to keep
Mistral as a second option.

`src/run_pilot.py` (`slurm/run_pilot.slurm`) still runs the 46-article pilot
through Llama-3.3-70B-Instruct before the full corpus run, not to compare it
against anything, but as a QA spot-check (`docs/SCORING_RUBRIC.md`) that the
extraction quality looks right across the psychology subfields before we spend
real GPU time on thousands of articles.
