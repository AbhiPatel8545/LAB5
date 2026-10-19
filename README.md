# Lab 05: Attention Against Recurrence

**Course:** AIGC 5500 – Advanced Deep Learning

## Introduction

In this lab, I compared attention and LSTM models on small tasks. I also tested different positional encodings, checked my multi-head attention implementation, and compared the training settings and running time of attention and LSTM.

## Files

- `lab05.py` – Python file that runs the experiments.
- `lab05_starter.py` – starter file provided for the lab, if required by the instructor.
- `README.md` – this file.
- `requirements.txt` – Python packages used for the project.
- `results/` – CSV files created when the script runs.

## How to Run

I ran the code on Windows using Python and a virtual environment.

Install the required packages:

```powershell
pip install torch numpy matplotlib
```

To run the experiments:

```powershell
python lab05_all_in_one.py
```

The script runs the experiments and saves CSV results in the `results` folder. The experiments can take some time on a CPU.

## Experiment 1: Attention Implementation

I tested my scaled dot-product attention and multi-head attention against PyTorch's attention implementation. I checked masked and unmasked attention, attention weight sums, masked weights, self-attention, causal attention, cross-attention, parameter counts, and permutation equivariance.

The attention tests passed. With `d_model=32`, the reported parameter count was 4,224 for 1, 2, 4, and 8 heads.

## Experiment 2: Delayed Recall

I compared attention and LSTM on sequence lengths 30, 100, and 400. I used three random seeds and 600 training updates. The chance accuracy for this task is 0.25. For the attention model, I also recorded the attention weight on the first time step.

In my run, attention reached 1.00 accuracy for all tested lengths and seeds. At length 30, the LSTM also reached 1.00 for all seeds. At length 100, one LSTM run was near chance at 0.252, while the other two reached 1.00. At length 400, the LSTM results were around 0.244–0.245, which is close to chance. This suggests that the LSTM had more difficulty remembering the first input for longer sequences in this experiment.

## Experiment 3: Positional Encoding

I compared three positional encoding settings: `none`, `learned`, and `sinusoidal`. I used three seeds for each setting.

For fixed-length training, the training length was 16 and the test lengths were 16 and 32. For mixed-length training, I used lengths 8, 16, and 24, and tested on lengths 16, 32, and 48. The chance accuracy is 0.50.

### Prediction before mixed-length training

Before running the mixed-length experiment, I expected sinusoidal positional encoding to work better on longer sequences because its position pattern can also be calculated for positions beyond the training length. I expected learned positions to work well on the training length but possibly perform worse on longer sequences. This was my prediction, not a guaranteed result.

### Results

For fixed-length training, all learned and sinusoidal runs reached 1.00 accuracy at length 16. At length 32, sinusoidal results were about 0.631–0.738, while learned results were about 0.604–0.625. The model without positional encoding stayed close to chance.

For mixed-length training, sinusoidal encoding reached 1.00 at length 16, about 0.893–0.929 at length 32, and about 0.593–0.742 at length 48. Learned encoding was strong at length 16 but dropped toward chance at length 48. The model without positional encoding stayed close to chance. In this run, sinusoidal encoding gave the best results on longer test sequences, although its length-48 results were not perfect.

## Experiment 4: Deep Stack and Timing

I tested a 12-block model for 400 updates, with a learning rate of `3e-3` and three seeds. I compared:

- Post-norm without warmup.
- Post-norm with 100-update warmup.
- Pre-norm without warmup.

Post-norm without warmup gave accuracy between 0.448 and 0.552. Post-norm with warmup and pre-norm without warmup both reached 1.00 for all three seeds in this run.

I also measured forward and backward time for an LSTM and one attention layer. The batch size was 32, the sequence lengths were 100, 200, 400, and 800, and I recorded the best of three measurements.

| Sequence length | LSTM time (seconds) | Attention time (seconds) |
|---:|---:|---:|
| 100 | 0.1711 | 0.0467 |
| 200 | 0.2240 | 0.0625 |
| 400 | 0.3432 | 0.1902 |
| 800 | 0.7055 | 0.9899 |

In my CPU run, attention was faster at lengths 100, 200, and 400, but slower at length 800. The timing can change depending on the computer and software setup.

## Conclusion

From these experiments, I found that attention performed well on the delayed recall task, while the LSTM struggled at longer sequence lengths in some runs. Sinusoidal positional encoding performed best on the longer mixed-length tests. Warmup improved the post-norm deep model in this run, and pre-norm also performed well. Attention was faster for shorter sequences in my timing test, but it was slower at length 800.

These results are from three seeds and small synthetic tasks, so they do not prove that the same results will happen for every dataset or computer.

## Output Files

After running the script, check the `results` folder for:

- `attention_parameters.csv`
- `recall_results.csv`
- `position_results.csv`
- `deep_stack_results.csv`
- `timing_results.csv`



