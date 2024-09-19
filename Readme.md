# Implementation of FLoRA (https://arxiv.org/abs/2312.05677)

## Overview
This project compares the performance of FLoRA (Fast Low-Rank Adaptation) and LoRA (Low-Rank Adaptation) in Transformer models. We evaluate the efficiency of both methods by measuring the throughput and latency across various model sizes and ranks.

## Experimental Results Interpretation
![Experimental Results](results/results.png)

The experimental results stored in the `results/results.png` file show a pattern similar to the throughput and latency experiments conducted in the FLoRA paper.

### Throughput Analysis
Similar to the paper, the throughput of FLoRA improves as the model size increases (small, medium, large). For large-size models, throughput is higher compared to medium and small models. Additionally, regardless of model size, applying FLoRA shows a trend where throughput decreases as the rank increases. When the model size is fixed, the rank at which the throughput of LoRA and FLoRA overlaps shifts to a higher rank as the model size increases. This pattern is consistent with the results from the paper.

### Latency Analysis
As seen in the paper, applying FLoRA results in an increase in latency as the rank increases, while applying LoRA maintains a relatively consistent level of latency.

In conclusion, the experiment implementation confirms that FLoRA significantly improves throughput and latency at smaller ranks, and it improves throughput even in larger models. However, the results in the paper show more pronounced differences, which is likely due to the paper’s use of larger models (1B, 3B, 15B) compared to the models used in this experiment.

## Experimental Reproduction Environment
The following describes how to set up the environment to reproduce the experiments:

### Experiment Structure
The experiments are divided into three main scripts:
- `layers.py`: Contains the definition of FLoRA, LoRA layers, and the Transformer decoder layers.
- `utils.py`: Includes utility functions for generating random data, measuring throughput and latency, and plotting results.
- `main.py`: The main script that runs the experiments, executes the models, and generates the result plots.

### Environment Setup
1. Python version: 3.10
2. Required packages:
    - torch==2.3.1
    - tqdm==4.66.4
    - matplotlib==3.9.1
    - numpy==1.26.3

After installing the packages, you can run the experiment with the following command (ensure you specify the `result_path` argument with your local path):

```bash
python main.py
