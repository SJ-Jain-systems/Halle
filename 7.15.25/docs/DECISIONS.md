# Project decisions

Source brief: `PLOS_ONE_PROJECT.pdf`, a study of demographic representativeness
in psychology research articles published in *PLOS ONE*, 2010 to 2026.

This doc writes down the three open decisions from the brief and why we went the
way we did. The code that implements them lives in `src/`, and how to actually
run it on Rivanna is in `docs/RUNNING_ON_RIVANNA.md`.

## 1. Data access: Solr for discovery, `allofplos` for full text

**Decision: pull the psychology article list from PLOS's Solr index
(`src/solr_client.py`, `src/build_index_solr.py`), and use the local `allofplos`
XML corpus only for the full text.**

This is basically the brief's *original* plan, but we only got here after trying
an allofplos-only approach and giving up on it. Worth writing down, because it's
a real data-quality gotcha:

1. **First try: allofplos only, parse the taxonomy from the local XML.** The plan
   was to sync the whole corpus once and read PLOS's subject taxonomy straight
   out of each article's JATS `<subj-group>` blocks (`src/jats_xml.py`,
   `src/build_corpus_index.py`), so the pipeline would be fully offline after the
   one-time download.
2. **Why it fell over:** once we checked against the real corpus (391k articles),
   it turned out the XML is missing the subject taxonomy for a big chunk of
   articles, and the chunk depends on the year. About 99% of 2015 and about 37%
   of 2013 only have a "Research Article" heading and no discipline tags at all
   (see `scripts/diagnose_2015.py`). So an XML-only scan quietly dropped almost
   all of 2015, which would gut the "middle" (2015 to 2019) stage of the
   analysis. No amount of better parsing fixes it, because the data just isn't in
   those files.
3. **Fix: use Solr to find the articles.** PLOS's Solr index has the real
   taxonomy for *every* article. `src/build_index_solr.py` asks it for every PLOS
   ONE research article from 2010 to 2026 tagged under Psychology
   (`subject:"Psychology"`, with the required `doc_type:full` and strict `fq`
   filters, see `scripts/test_solr.py`), reads the subfield out of the subject
   *paths*, and points each DOI at its local XML file for the full text. Gap-free
   and consistent across all the years.

The subject metadata (brief item 4.c.i: `subject` / `subject_level_1`) comes from
Solr's subject paths; the full text (Methods/Participants) still comes from the
local XML via `src/jats_xml.py::get_extraction_text()`, which is there even for
the articles with no taxonomy.

Net effect: one network step (the Solr enumeration, on the login node, about 10
to 20 minutes instead of the roughly 1.5 hour XML scan) plus the one-time corpus
download, and everything after that reads local XML. The old XML-scan code
(`src/build_corpus_index.py`, `slurm/build_index.slurm`) is still around for
reference but it's superseded.

## 2. Pilot sample

**Decision: 2 articles per psychology subfield, across *all* the subfields PLOS
actually tags (not a hardcoded five), drawn at random with a fixed seed so it's
reproducible.**

The brief (item 3.c) named five subfields: social, cognitive, developmental,
clinical, and quantitative psychology. But once we checked against the real
corpus, PLOS's taxonomy doesn't even use "Quantitative psychology" as a term, and
*does* use several the brief didn't mention (Experimental psychology,
Psychometrics, and others). Rather than force the brief's list onto a taxonomy
that doesn't match it, the pipeline is taxonomy-driven: it grabs every article
tagged under the Psychology node and records whatever subfield(s) PLOS gave it
(see decision #1 and `jats_xml.get_psychology_subfields`). The pilot then
stratifies over whatever subfields the index actually has.

Why stratify instead of just pooling everything and sampling at random: a uniform
draw risks a high-volume subfield (social or cognitive, say) crowding out the
rest, and the whole point of the pilot is to check the model's extraction quality
*across* subfields before we spend GPU time on the full run. Two per subfield
guarantees coverage. (That's why the pilot is 2×N articles for N subfields, not a
fixed 10. The brief's "10" assumed exactly 5 subfields.)

Implementation: `src/sample_articles.py`, the `stratified_sample()` function (it
works out the subfields from the index and skips any with fewer than 2 articles),
reading from `data/corpus_index.csv` (no network). The seed defaults to `42`;
override it with `--seed`.

## 3. Model choice

**Decision: `meta-llama/Llama-3.3-70B-Instruct`. Final, this is the only model
the pipeline runs.**

The brief gave two candidates and asked us to pick one:

| | Mistral-7B-Instruct-v0.3 | Llama-3.3-70B-Instruct |
|---|---|---|
| Parameters | 7B | 70B |
| Context window | 32K | 128K |
| Structured extraction / instruction-following accuracy | Noticeably weaker on multi-field JSON extraction and on numeric reasoning buried in prose | Clearly stronger, and better at not hallucinating percentages or dropping a demographic subgroup |
| Compute/cost | Runs on a single consumer GPU or free-tier hosted inference | Needs several GPUs, which is a real constraint on shared infrastructure but not on a dedicated HPC allocation |

Llama-3.3-70B-Instruct wins on the thing that actually matters here: a missed or
made-up demographic percentage feeds straight into the representativeness
analysis, which is the whole point of the project, so extraction accuracy is what
we care about most. The only reason to lean toward Mistral-7B would be compute
cost, and running on Rivanna (a dedicated GPU allocation, no per-token API cost,
no rate limits) takes that off the table. There's no real tradeoff left, so
there's no reason to keep Mistral around as a second option.

`src/run_pilot.py` (`slurm/run_pilot.slurm`) still runs the 46-article pilot
through Llama-3.3-70B-Instruct before the full corpus run, not to compare it
against anything, but as a QA spot-check (`docs/SCORING_RUBRIC.md`) that the
extraction quality looks right across the psychology subfields before we spend
real GPU time on thousands of articles.
