# Movie Recommender: Hybrid Content-Based + Collaborative Filtering System

A movie recommendation engine built from scratch, combining content-based
filtering (what a movie is about) with collaborative filtering (who liked
what) into a single hybrid system. Built on the MovieLens small dataset
and Kaggle's TMDB metadata.

## Overview
Most recommenders lean on one signal alone: either "movies similar
to this one" (content-based) or "users like you also liked..."
(CF). This project combines both, and empirically
validates that the combination outperforms either approach individually.

## Why Hybrid?
Early EDA revealed two structural problems that neither method alone
can fully solve:

- **The ratings matrix is 98.3% sparse** (610 users × 9,724 movies,
  100,836 ratings). A quarter of all movies have only a single rating;
  CF has almost nothing to learn from for the long tail of the catalog.
- **Genre tags are too coarse to differentiate movies.** The average
  movie carries only ~2 genres, and Drama/Comedy dominate the catalog;
  two wildly different films can share identical genre tags.

Content-based filtering fills the gap where CF has no data (new/rarely
rated movies); CF captures taste patterns that no amount of plot/genre
metadata can reveal.

## How It Works
**Content-based engine:** builds a "metadata soup" per movie (overview +
genres + keywords + director), vectorizes it with TF-IDF (term frequency-inverse document frequency), 
and computes cosine similarity across all 9,525
movies with valid metadata.

**Collaborative filtering engine:** trains an SVD matrix factorization
model (via `surprise`), tuned via grid search
(n_factors=150, n_epochs=30, lr_all=0.01, reg_all=0.1).

**Hybrid combination:** for a given user and candidate movie, both
engines produce a score, each normalized to 0–1, then combined as
`0.6 * CF_score + 0.4 * content_score`. Movies with no content metadata 
fall back to CF-only scoring.

## Key Technical Decisions
- **Director names are underscore-joined** ("John_Lasseter") before
  vectorization, so TF-IDF treats them as one token instead
  of splitting into generic words that could coincidentally overlap
  with unrelated names.
- **Content scoring averages only the top-10 most similar liked movies**,
  Averaging over a prolific rater's entire history
  (some users have 200+ liked movies) diluted genuine signal down to
  near-zero; capping at the top-10 restored it.
- **Content score normalization uses an empirically-derived range**
  (0.0–0.25), not the theoretical 0–1 bound. Real similarity scores
  rarely approach 1.0 in practice, so normalizing against the
  theoretical max made content signal look artificially weak next to
  CF's cleaner 0.5–5.0 scale.
- **Movies with no metadata mapping (~2% of the catalog) are excluded**
  from hybrid recommendations entirely, rather than penalized. They
  got an unfair ranking advantage by skipping the moderating
  content score.

## Results
**RMSE (collaborative filtering, 5-fold CV):** 0.8621 after tuning
(from an untuned baseline of 0.8807)

**Precision@10, hybrid vs. individual methods** (25-user independent
sample):

| Method | Precision@10 |
|---|---|
| Hybrid | 0.132 |
| Content-only | 0.056 |
| CF-only | 0.020 |

The hybrid approach outperformed CF-only by ~6.6x and content-only by
~2.4x. Interestingly, content-only outperformed CF-only here — evidence
the two signals capture genuinely different, complementary information
rather than one simply dominating the other.

## Known Limitations
- **Tone-blind:** TF-IDF catches vocabulary/concept overlap but not
  mood: e.g. the horror film *Child's Play* scores meaningfully
  similar to *Toy Story* purely due to shared "toy" vocabulary.
- **No fuzzy title matching:** misspelled searches return "no match"
  rather than a best guess.
- **Not optimized for real-time serving:** the current candidate-scoring
  approach takes several seconds per user (naive loop over ~9,500
  candidates); a production version would need precomputation or
  vectorized scoring.
- **~2% of the catalog is unrecommendable** via the hybrid function due
  to missing metadata mappings.

## Tech Stack
Python, pandas, scikit-learn (TF-IDF, cosine similarity), `surprise`
(SVD), Jupyter

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Data: [MovieLens small dataset](https://grouplens.org/datasets/movielens/)
and [Kaggle's "The Movies Dataset"](https://www.kaggle.com/rounakbanik/the-movies-dataset)
(not included — place in `data/`, see notebook for expected filenames).

## Next
Phase 2 wraps this engine in a full-stack web app — React frontend,
Flask/FastAPI backend, user accounts, and real onboarding (rate a few
movies, get real recommendations).