# Learning Molecular Representation in a Cell
================================================================

## Requirements

This code was developed and tested with Python 3.11.7, PyTorch 2.1.0+cu118, and torch-geometric 2.5.2.
All dependencies are specified in the `requirements.txt` file.

## Usage

```
# pretrain
python main.py --model-path "ckpt/pretrain.pt" --lr 1e-4 --wdecay 1e-8 --batch-size 3072

# finetune
python main.py --model-path ckpt/pretrain.pt --dataset finetune-chembl2k
```

