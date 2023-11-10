import logging
import argparse
from transformers import AutoImageProcessor, AutoModelForImageClassification
from time import perf_counter
import numpy as np
import torch
import csv
import os
from datasets import load_dataset

# Set up logging
logger = logging.getLogger(__name__)

def generate_sample_inputs(processor):
  dataset = load_dataset("huggingface/cats-image")
  image = dataset["test"]["image"][0]
  embeddings = processor(image,return_tensors="pt")
  embeddings = {k: v.to("cuda") for k, v in embeddings.items()}
  return tuple(embeddings.values())
  

def measure_latency(model, input):
    latencies = []
    # warm up
    for _ in range(10):
        _ = model(*input)
    # Timed run
    for _ in range(100):
        start_time = perf_counter()
        _ =  model(*input)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies,95)
    time_p99_ms = 1000 * np.percentile(latencies,99)
    return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms, "time_p95_ms": time_p95_ms, "time_p99_ms": time_p99_ms}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_neuron", action="store_true")
    parser.add_argument("--model_id", type=str)  
    parser.add_argument("--instance_type", type=str)  
    parser.add_argument("--sequence_length", type=int, default=None)
    
    # neuron specific args
    parser.add_argument("--num_neuron_cores", type=int, default=1)
    known_args, _ = parser.parse_known_args()  
    return known_args

def compile_model_inf1(model, payload, num_neuron_cores):
    os.environ['NEURON_RT_NUM_CORES'] = str(num_neuron_cores)
    import torch.neuron
    return torch.neuron.trace(model, payload)

def compile_model_inf2(model, payload, num_neuron_cores):
    # use only one neuron core
    os.environ["NEURON_RT_NUM_CORES"] = str(num_neuron_cores)
    import torch_neuronx
    return torch_neuronx.trace(model, payload)


def main(args):
  print(args)

  # benchmark model
  benchmark_dict = []
  # load processor and  model
  processor = AutoImageProcessor.from_pretrained(args.model_id)
  model = AutoModelForImageClassification.from_pretrained(args.model_id, torchscript=True)
  model.to("cuda")
  
  # generate sample inputs
  payload = generate_sample_inputs(processor)


  res = measure_latency(model, payload)
  print(res)
  benchmark_dict.append({**res,"instance_type": args.instance_type})    
  
  # write results to csv
  keys = benchmark_dict[0].keys()
  with open(f'results/benchmmark_{args.instance_type}_{args.model_id.split("/")[-1].replace("-","_")}.csv', 'w', newline='') as output_file:
      dict_writer = csv.DictWriter(output_file, keys)
      dict_writer.writeheader()
      dict_writer.writerows(benchmark_dict)

if __name__ == "__main__":
  main(parse_args())
  
  
# python3 scripts/inference_transformers_vision_gpu.py --model_id google/vit-base-patch16-224 --instance_type g5.2xlarge