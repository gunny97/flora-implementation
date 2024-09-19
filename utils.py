import matplotlib.pyplot as plt
import time
import torch
import os
import numpy as np
from tqdm import tqdm

def plot_results(rank_list, results, output_file='results/throughput_latency_results.png'):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for model_size, result in results.items():
        if 'small-size' in model_size:
            plt.plot(rank_list, result['lora_throughputs'], label=f'LoRA Throughput {model_size}', marker='o', color='orange', linestyle='-')
            plt.plot(rank_list, result['flora_throughputs'], label=f'FLoRA Throughput {model_size}', marker='o', color='blue', linestyle='-')
        elif 'medium-size' in model_size:
            plt.plot(rank_list, result['lora_throughputs'], label=f'LoRA Throughput {model_size}', marker='o', color='orange', linestyle='--')
            plt.plot(rank_list, result['flora_throughputs'], label=f'FLoRA Throughput {model_size}', marker='o', color='blue', linestyle='--')
        elif 'large-size' in model_size:
            plt.plot(rank_list, result['lora_throughputs'], label=f'LoRA Throughput {model_size}', marker='o', color='orange', linestyle=':')
            plt.plot(rank_list, result['flora_throughputs'], label=f'FLoRA Throughput {model_size}', marker='o', color='blue', linestyle=':')

    plt.xlabel('Rank')
    plt.ylabel('Throughput (Tokens/s)')
    plt.title('Throughput vs. Rank')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rank_list, results["large-size"]['lora_latencies'], label='LoRA Latency 1B', marker='o', color='orange')
    plt.plot(rank_list, results["large-size"]['flora_latencies'], label='FLoRA Latency 1B', marker='o', color='blue')
    plt.xlabel('Rank')
    plt.ylabel('Latency (Seconds per Output Token)')
    plt.title('Latency vs. Rank')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def generate_random_data(batch_size, num_samples, vocab_size, min_length=100, max_length=2000):
    data = []
    for _ in range(num_samples):
        length = torch.randint(min_length, max_length, (1,)).item()
        sample = torch.randint(0, vocab_size, (batch_size, length))
        data.append(sample)
    return data

def measure_throughput_latency(model, data, device, requests_per_second=8):
    model.to(device)
    model.eval()
    total_time = 0
    total_tokens = 0
    latency_per_batch = []

    with torch.no_grad():
        for i in tqdm(range(len(data)), desc="Processing Batches"):
            batch = data[i].to(device)
            start_time = time.time()
            _ = model(batch, batch)
            end_time = time.time()
            
            batch_time = end_time - start_time
            latency_per_batch.append(batch_time)
            total_time += batch_time
            total_tokens += batch.size(0) * batch.size(1)
            
            time.sleep(1.0 / requests_per_second)
    
    throughput = total_tokens / total_time
    latency = sum(latency_per_batch) / len(latency_per_batch)
    
    return throughput, latency
