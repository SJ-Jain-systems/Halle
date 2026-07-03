# Decisions

The three decisions the brief left open, and what we did about each.

## 1. Where the data comes from

We use PLOS's search engine (Solr) to find the psychology papers, and the
downloaded article files (allofplos) only for the full text. Search engine for
discovery, local files for reading.

We did not start there. The first plan was to skip the search engine entirely:
download every PLOS article and read the subject tags straight out of the files.
That failed for a reason worth knowing.

When we counted psychology papers per year, 2015 came back with 12 papers, versus
about 2,000 in the years on either side. It turned out that for roughly 99% of
2015 articles, and about 37% of 2013, the downloaded files have no subject tags
at all. Just a "Research Article" heading and nothing else. PLOS classified those
papers on its website but never wrote the tags into the files. You cannot read a
tag that isn't there.

The search engine does have the tags for every article. So we switched: use it to
build the list of which papers qualify, and keep the local files for the one
thing they always contain, the text. This is the split the original brief
described, before we tried the shortcut.

Two smaller things we hit on the way:
- The tag container was renamed across PLOS's history, from "Discipline" to
  "Discipline-v3". The first parser only matched the old name and found nothing.
  We now accept both.
- The brief named five subfields, but PLOS has no "Quantitative psychology" tag
  and uses several the brief didn't list. So we take whatever subfield PLOS
  actually assigns, which gives 23 subfields instead of 5.

## 2. The pilot sample

Before running the model on all 58,000 papers, we test it on a small batch and
grade it by hand.

We draw 2 papers from each subfield, not 46 at random. A random draw would be
dominated by the big subfields and might skip the rare ones. Two per subfield
covers the full range. The draw is seeded, so anyone re-running it gets the same
batch. The result is 46 papers across 23 subfields.

## 3. Which model

Llama-3.3-70B, over the smaller Mistral-7B.

Accuracy is what matters. A missed or invented demographic number goes straight
into the final trends and skews them. The bigger model is more reliable at this
kind of structured extraction. It costs more to run, but on Rivanna compute isn't
the constraint, so cost doesn't decide it. The pilot is there to confirm the
model is good enough before the full run.
