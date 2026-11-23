# CO5135 Big Data: Large-scale Recommender System with LSH

Mini-project for CO5135 Big Data (HCMUT, Semester 1, 2026-2027).
Student: Pham Thi Thai Minh (2570757).

## Problem

Item-based collaborative filtering needs the nearest neighbors of every item.
Brute-force search costs O(n²) and does not scale as data grows.
This project uses Locality Sensitive Hashing (LSH) in Spark to find similar items
approximately, and compares accuracy and runtime against brute-force.

## Method

1. Train ALS (Spark MLlib) to get latent vectors for items.
2. Find top-K similar items with brute-force (baseline) and with LSH
   (`BucketedRandomProjectionLSH` on normalized vectors).
3. Recommend movies from neighbors of items a user liked.
4. Evaluate: RMSE (ALS), Precision@K (recommendation), Recall@K of LSH vs brute-force,
   runtime on 1M, 5M, 10M, 25M ratings.

## Dataset

MovieLens 25M (https://grouplens.org/datasets/movielens/25m/).
Data is downloaded automatically in `00_prepare_data` and is not stored in this repo.

| Subset | Ratings | Users | Movies | Sparsity |
|---|---|---|---|---|
| 1M | 999,935 | 6,580 | 19,991 | 99.24% |
| 5M | 4,999,764 | 32,665 | 35,730 | 99.57% |
| 10M | 9,999,663 | 65,080 | 45,417 | 99.66% |
| 25M | 25,000,095 | 162,541 | 59,047 | 99.74% |

Subsets are sampled by user (seed 42) and nested (1M in 5M in 10M in 25M).
Train/test split: 80/20 (seed 42).

## Environment

- Kaggle Notebooks, CPU, 4 cores, about 30 GB RAM
- PySpark 4.0.4 (`local[*]`, driver memory 20g)

## Notebooks

| Notebook | Content |
|---|---|
| `00-prepare-data.ipynb` | Download data, CSV to Parquet, create subsets, train/test split |
| `01-eda.ipynb` | Rating distribution, long-tail analysis |
| `02-als-bf-lsh.ipynb` | ALS, brute-force kNN, LSH (Spark BRP and SimHash banding) on the 1M subset |
| `03-scalability.ipynb` | Brute-force vs LSH runtime from 1M to 25M |
| `04-seminar-demo.ipynb` | MinHash, SimHash, banding S-curve demo |

## How to run

1. Open a notebook on Kaggle, set Accelerator = None and Internet = On.
2. Run `00_prepare_data` first, then add its output as Input to the other notebooks.
3. Run All.

## Results

(To be updated)
