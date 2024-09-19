import time
import argparse

import torch
import torch.nn as nn

from layers import LoRATransformerDecoderLayer, FLoRATransformerDecoderLayer
from utils import generate_random_data, measure_throughput_latency, plot_results


def run_experiment(rank_list, model_sizes, num_samples, vocab_size, batch_size, num_layers, dim_feedforward, device):
    results = {}
    for model_size, (d_model, nhead) in model_sizes.items():
        print(f"Model size: {model_size}")
        src_batches = generate_random_data(batch_size, num_samples, vocab_size)
        lora_throughputs, lora_latencies = [], []
        flora_throughputs, flora_latencies = [], []

        for rank in rank_list:
            print(f"Rank: {rank}")
            lora_decoder_layer = LoRATransformerDecoderLayer(d_model, nhead, dim_feedforward, rank=rank, num_adapters=batch_size)
            lora_model = nn.TransformerDecoder(lora_decoder_layer, num_layers=num_layers)
            
            flora_decoder_layer = FLoRATransformerDecoderLayer(d_model, nhead, dim_feedforward, rank=rank, batch_size=batch_size)
            flora_model = nn.TransformerDecoder(flora_decoder_layer, num_layers=num_layers)

            lora_model.to(device)
            flora_model.to(device)

            lora_throughput, lora_latency = measure_throughput_latency(lora_model, src_batches, device)
            print('LoRA Done!')
            flora_throughput, flora_latency = measure_throughput_latency(flora_model, src_batches, device)
            print('FLoRA Done!')
            
            lora_throughputs.append(lora_throughput)
            lora_latencies.append(lora_latency)
            flora_throughputs.append(flora_throughput)
            flora_latencies.append(flora_latency)

        results[model_size] = {
            'lora_throughputs': lora_throughputs,
            'lora_latencies': lora_latencies,
            'flora_throughputs': flora_throughputs,
            'flora_latencies': flora_latencies
        }

    return results

def main(args):
    device = torch.device(args.device)
    vocab_size = args.vocab_size
    batch_size = args.batch_size
    num_layers = args.num_layers
    dim_feedforward = args.dim_feedforward
    num_samples = args.num_samples
    rank_list = [1, 2, 4, 8, 16]

    model_sizes = {
        'small-size': (args.d_model, args.num_heads),
        'medium-size': (args.d_model * 3, args.num_heads * 3),
        'large-size': (args.d_model * 15, args.num_heads * 15)
    }

    results = run_experiment(rank_list, model_sizes, num_samples, vocab_size, batch_size, num_layers, dim_feedforward, device)
    plot_results(rank_list, results, output_file=args.result_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default='/home/keonwoo/workspace/flora/results/results.png')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--vocab_size', type=int, default=100)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=256)

    args = parser.parse_args()
    main(args)