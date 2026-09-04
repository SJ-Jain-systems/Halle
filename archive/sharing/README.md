# Sharing folder

A copy of the basis of the project to understand it without reading
the whole repo. The scripts in `src/` have plain notes added at the top and
through the code. The code itself is unchanged from the working versions.

## Read in this order

1. `project_report.qmd` is the writeup. Start here. It covers the question, the
   decisions, the data problems I hit and how I fixed them, and where things
   stand. Render it with `quarto render project_report.qmd --to html`, or open
   it in RStudio.

2. `src/` is the code, in roughly the order it runs:
   - `solr_client.py` finds the psychology papers through PLOS's search engine
   - `build_index_solr.py` writes the master list of 58,467 papers
   - `jats_xml.py` reads metadata and text out of the article files
   - `sample_articles.py` draws the pilot test batch
   - `extract_demographics.py` the model instruction and the output checker
   - `model_backend.py` runs the model on the GPUs
   - `run_pipeline.py` the full run, split into parallel restartable chunks
   - `analyze_trends.py` turns the results into reporting-rate trends

3. `docs/` is the reference material:
   - `DECISIONS.md` the three main decisions and why
   - `RUNNING_ON_RIVANNA.md` how to run the pipeline on the cluster
   - `SCORING_RUBRIC.md` how I grade the model on the pilot

These are copies. The live code and full history are in the main repo but tbh the main repo is a bit of a mess and I haven't really gotten past step 4 without debugging. The tests work, but I haven't put them into practice yet. 
