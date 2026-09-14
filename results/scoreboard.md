# Scoreboard

Config: `configs/launch_100k.yaml` · graph `cb5k` · metric `final_val` (nats/char). 
Same data, same split, same context, same training loop for every row.

Measured on our split: unigram 3.347, bigram 2.482.
External reference: nanoGPT `train_shakespeare_char` best val ~1.47 (verify + cite at launch).

| Model | Params | Val loss (mean) | Range | Seeds | Notes |
|---|--:|--:|---|--:|---|
| degree_preserving | 578,197 | 1.613 | 1.606–1.630 | 5 | |
| real | 578,197 | 1.609 | 1.603–1.619 | 5 | |
