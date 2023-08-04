# How To Build A Neural Search Engine 

## Overview

This repository uses [Vespa Vector Database](https://vespa.ai/) to build a multilingual multimodal vector search engine using 
publicly available pretrained models, as well as open-source datasets.

## Prerequisites

- Python 3.6 or above
- Sentence Transformers
- Open Clip
- Vespa learntorank

## Installation

1. Clone the repository: 
```
https://github.com/new/import
```

2. Install the required packages: 
```
pip install -r requirements.txt
```

3. Install Docker Desktop: 
```
https://docs.docker.com/engine/install/
```

## Datasets
We use publicly available datasets to evaluate the search engine. 

#### Multi30k
A dataset to stimulate multilingual multimodal research. 
We use test set that contains 1000 Flickr image-caption pairs.
Captions are in EN, FR, DE languages. 

#### Roco
Radiology Objects in COntext (ROCO) dataset, a large-scale medical and multimodal imaging dataset.
We use test set which contains 8000 image-caption pairs (English only).

#### Coco
The Microsoft Common Objects in COntext (MS COCO) dataset contains human generated captions for images. 
We use the test set which contains 5000 image-caption pairs (English only).


**Note:** The repository assumes you have the datasets stored in a directory named `data/`. 
Please make sure to download the datasets and organize them accordingly.

```.
ir-search/baseline/data
├── roco_test
│   ├── images
│   │   ├── ROCO_00001.jpg
│   │   ├── ROCO_00006.jpg
│   │   └── ... 
│   ├── captions.txt
│   ├── image_ids.txt
│   └── semantic_tags.txt
├── multi30k_test
│   ├── test_2017-flickr-images
│   │   ├── 8085680939_6c56b595ea.jpg
│   │   ├── 8419925347_32ff7386ae.jpg
│   │   └── ... 
│   ├── test_2017_flickr.en
│   ├── test_2017_flickr.fr
│   ├── test_2017_flickr.de
│   └── test_2017_flickr.txt
└── coco2014_test
    ├── images
    │   ├── COCO_val2014_000000391895.jpg
    │   ├── COCO_val2014_000000060623.jpg
    │   └── ... 
    ├── captions.txt
    └── image_ids.txt
 ```

## Pretrained Models
Current pretrained models are all variations of CLIP (Contrastive Language-Image Pre-training) model. 
[Visit OpenAI's website](https://openai.com/clip)

#### OpenAI CLIP
- image encoder: OpenAI "clip-ViT-B-32" trained on 400M (image, english caption) pairs
- text encoder: SBERT multilingual DistilBERT that maps text in 50+ languages to
  the original "clip-ViT-B-32" vector space.

#### LAION CLIP
Open source implementation of OpenAI's CLIP from [LAION community](https://openai.com/clip), 
called OpenCLIP. Currently, we use first multilingual clip trained with openclip: 
- ViT-B/32 xlm roberta base trained on LAION-5B: multilingual CLIP-filtered image-text pairs. 

#### Microsoft BiomedCLIP
Microsoft biomedical VLP pretrained on PMC-15M, 
a dataset of 15 million figure-caption pairs extracted from biomedical research articles in PubMed Central, 
using contrastive learning. Currently, we use
- BiomedCLIP-PubMedBERT_256-vit_base_patch16_224


## Metrics
Default metric to measure the performance is recall.
Other available metrics to select: 
- MRR: mean reciprocal rank
- NDCG: normalized discounted cumulative gain
- match ratio


## Usage 

To run the experiment, use the following command:
```
python retrieval_evaluation.py --model_name openai_clip --dataset roco --metrics recall
```
You can switch the model and dataset as required. 

**Sample single-process running code:**
```
python retrieval_evaluation.py \
    --device="cpu" \
    --seed=1 \
    --model_name="open_ai" \
    --app_name="text2image" \
    --search_type="vector" \
    --dataset="flickr_english" \
    --metrics="recall" \
    --preprocess=False \
    --save_results=False \
    --output_dir="./outputs/"
```

## Results

The results of the experiment will be saved in the `outputs/` directory.

## Retrieval Evaluation
Below is a comparison of the models' performance on the ROCO and Flickr datasets:

### 1. Text to Image Retrieval
Sending text query on image data (cross modal search)

#### A) Relevancy or Quality Assessment
Measuring how much the returned results are relevant to the query 

##### Recall@1

| Model                 | OpenAI CLIP | LIAON CLIP |   Microsoft BiomedCLIP   |
|-----------------------|:-----------:|:----------:|:------------------------:|
| Roco8k-English        |    0.005    |   0.012    |          0.349           |
| MultiFlickr1k-English |    0.557    |   0.669    |          0.033           |
| MultiFlickr1k-French  |    0.465    |   0.607    |          0.001           |
| MultiFlickr1k-Dutch   |    0.414    |   0.588    |          0.003           |
| Coco5k-English        |    0.244    |   0.362    |          0.010           |


<br>

#### B) Latency or Speed Assessment
Measuring how fast the results are returned by the search engine

| Model               | OpenAI CLIP | LIAON CLIP |  Microsoft BiomedCLIP   |
|---------------------|:-----------:|:----------:|:-----------------------:|
| Encode one pair (s) |    0.090    |   0.270    |          0.382          |
| Query time (s)      |    0.023    |   0.210    |          0.252          |

**Note:** These numbers are the averages over 5000 queries but only one dataset.
The absolute values change based on dataset and machine. However, the order of magnitude is more reliable.

Test done on MacBook Pro machine:  
- Processor: 2.6 GHz 6-Core Intel Core i7 
- Memory: 32 GB 2667 MHz DDR4
- Graphics: AMD Radeon Pro 5300M 4 GB

<br>

### 2. Image to Text Retrieval
TODO

<br>

## Acknowledgments

This code is based on the following repositories:

- SBERT CLIP (https://github.com/UKPLab/sentence-transformers/tree/master)
- LIAON CLIP (https://github.com/mlfoundations/open_clip)
- Microsoft BiomedCLIP (https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- COCO test set (https://huggingface.co/datasets/nlphuji/mscoco_2014_5k_test_image_text_retrieval/viewer/TEST/test?row=0)
- ROCO test set (https://huggingface.co/datasets/MedIR/roco/viewer/MedIR--roco/test?row=0)
- Multi30k test set (https://github.com/multi30k/dataset)


