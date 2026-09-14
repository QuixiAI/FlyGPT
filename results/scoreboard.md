# Scoreboard

Config: `configs/launch_100k.yaml` · graph `cb5k` · metric `final_val` (nats/char). 
Same data, same split, same context, same training loop for every row.

Measured on our split: unigram 3.347, bigram 2.482.
External reference: nanoGPT `train_shakespeare_char` best val ~1.47 (verify + cite at launch).

| Model | Params | Final val (mean) | Range | Best val (mean) | Range | Seeds | Notes |
|---|--:|--:|---|--:|---|--:|---|
| degree_preserving | 578,197 | 1.613 | 1.606–1.630 | 1.591 | 1.582–1.606 | 5 | same neurons/degrees, wiring scrambled |
| gru | 577,545 | 1.606 | 1.598–1.615 | 1.518 | 1.508–1.528 | 2 | GRU, parameter-matched; overfits after its best step |
| real | 578,197 | 1.609 | 1.603–1.619 | 1.587 | 1.578–1.599 | 5 | real fly wiring, edges trained |
| rnn | 578,055 | 2.021 | 2.016–2.026 | 1.578 | 1.574–1.582 | 2 | dense tanh RNN, parameter-matched; overfits after its best step |
