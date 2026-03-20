#!/bin/sh

mkdir -p data

uv run gdown "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.csv" -O data/qm7.csv
uv run gdown "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv" -O data/Lipophilicity.csv

uv run python preprocessing/clean.py
uv run python preprocessing/splits.py