#!/usr/bin/env python
"""Download Tiny Shakespeare, fix the 90/10 split, hash the corpus, measure unigram/bigram references.
Writes data/shakespeare/split.json (committed).   python prepare_data.py configs/launch.yaml"""
import argparse, json
from flygpt import Config
from flygpt.data import prepare

ap = argparse.ArgumentParser()
ap.add_argument("config", nargs="?", default="configs/launch.yaml")
args = ap.parse_args()
info = prepare(Config.load(args.config).dataset)
print(json.dumps({k: v for k, v in info.items() if k != "vocab"}, indent=2))
