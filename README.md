[![DOI](https://zenodo.org/badge/660734586.svg)](https://zenodo.org/badge/latestdoi/660734586)
# Calculon-MoE - An extension of Calculon to support the modeling of Mixture of Experts (MoE) Architectures

### Setup with Conda 
If you don't have conda available:
```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
```
### Assuming the base conda has been activated already
``` sh
conda env create -f environment.yml --name calculon-moe
conda activate calculon-moe
```

## Running dense LLM Example (calculon)

### Performance output of a single run for the specified model and system configs

``` sh
$ PYTHONPATH=. ./bin/calculon llm models/megatron-1T.json examples/megatron_1T_training_4096_original.json systems/a100_80g.json -
```

### Search for the best config sweeping different system setups under constraints for the input model

``` sh
$ PYTHONPATH=. ./bin/calculon llm-optimal-execution models/megatron-1T.json 5128 2520 float16 systems/a100_80g.json output.json -m
```

## Running MoE LLM Example (calculon-MoE)

### Performance output of a single run for the specified model and system configs

Run a single calculon training modeling with GPT-like 1.8T MoE Transformer model (models/gpt-1.8T.json) and 4096 H100_80g GPUs (systems/H100_80g.json) used. The execution script (examples/gpt_1.8_training_4096.json) includes the details of the parameters (i.e., TP/DP/PP/EP/ES, etc) used for the execution.

``` sh
$ PYTHONPATH=. ./bin/calculon llm models/gpt-1.8T.json examples/gpt_1.8_training_4096.json systems/H100_80g_sxm.json -
```

### Running MoE LLM optimal search for the best config (calculon-MoE)

Run a system execution optimizer for searching the space for GPT-like 1.8T Transformer. The following example searches the parallelization technique for 4096 H100 GPUs, and the Batch Size is 2048, which is specified internally in the calculon/llm/optimal_execution_MoE file:

``` sh
$ PYTHONPATH=. ./bin/calculon llm-optimal-execution-moe models/gpt-1.8T_2.json 4096 2048 float16 systems/H100_80g_sxm.json output_gpt-1.8T_4096_2048.json -moe 16
```

## Publications

* Scaling Intelligence: Designing Data Centers for Next-Gen Language Models \
Jesmin Jahan Tithi, Hanjiang Wu, Avishaii Abuhatzera, Fabrizio Petrini \
[Paper](https://arxiv.org/abs/2506.15006)

* Calculon: A Methodology and Tool for High-Level Co-Design of Systems and Large Language Models\
Mikhail Isaev, Nic McDonald, Larry Dennison, Richard Vuduc\
[Paper](https://dl.acm.org/doi/pdf/10.1145/3581784.3607102)

* Scaling Infrastructure to Support Multi-Trillion Parameter LLM Training\
Mikhail Isaev, Nic McDonald, Richard Vuduc\
[Paper](https://openreview.net/pdf?id=rqn2v1Ltgn0)
