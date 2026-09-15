## Environment

Developed with Python 3.11.

```
torch 2.7.0
transformers 4.57.3
datasets 4.4.1
numpy 1.26.4
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```


### Category granularity (`--granularity`)

| Flag | Categories | Example |
|------|------------|---------|
| `EA` | `ENTITY#ATTRIBUTE` | `FOOD#QUALITY` |
| `E`  | `ENTITY`  | `FOOD` |

## Training
Using the rest15 dataset as a running example,
```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
    --train_path data/rest15/rest15_Train.json \
    --test_path  data/rest15/rest15_Test.json \
    --dataset_name rest15 \
    --output_base run \
    --granularity EA \
    --model_name google/flan-t5-large \
    --gamma 0.6
```

### Outputs

Each run writes to `${output_base}/${granularity}/${dataset_name}/`

## Layout

```
EmoCDA/
├── data/
│   ├── rest15/
│   ├── rest16/
│   ├── lap15/
│   └── lap16/
├── utils/
│   ├── __init__.py
│   ├── data.py          
│   ├── metrics.py            
│   └── decoding.py             
├── model/
│   ├── __init__.py
│   ├── dual_decoder.py           
│   └── cross_decoder_attention.py
├── training/
│   ├── __init__.py
│   ├── collator.py            
│   ├── trainer.py             
│   └── pipeline.py               
├── run.py                        
└── requirements.txt
```
