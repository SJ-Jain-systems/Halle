# Sharing folder

A self-contained snapshot for colleagues who want to understand this project
without reading the whole repository. Everything here is a copy of files from
the main project, with one difference: the scripts in `src/` have extra
plain-English annotations added at the top and inline, aimed at a reader who is
not steeped in the code. The actual code is unchanged (verified token-for-token
identical to the working versions).

## What to read first

1. `project_report.qmd` — the narrative report. Start here. It explains the
   research question, the design decisions, the data problems we hit and how we
   fixed them, and where the project stands. Render it to HTML or PDF with
   Quarto (`quarto render project_report.qmd --to html`) or open it in RStudio.

2. `src/` — the annotated scripts, in roughly the order the pipeline runs them:
   - `solr_client.py` — find the psychology papers via PLOS's search engine
   - `build_index_solr.py` — write the master list of 58,467 papers
   - `jats_xml.py` — read metadata and full text out of the article files
   - `sample_articles.py` — draw the stratified pilot test batch
   - `extract_demographics.py` — the model instruction and the strict output checker
   - `model_backend.py` — run the model on the cluster's GPUs
   - `run_pipeline.py` — the full run, split into parallel restartable chunks
   - `analyze_trends.py` — turn the results into reporting-rate trends

3. `docs/` — supporting documents:
   - `DECISIONS.md` — the three key decisions with full reasoning
   - `RUNNING_ON_RIVANNA.md` — how to actually run the pipeline on the cluster
   - `SCORING_RUBRIC.md` — how the model's pilot accuracy is graded

## Note

These are copies. The live, running versions and the full commit history live in
the main repository (`SJ-Jain-systems/Halle`).
