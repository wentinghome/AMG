# Attend, Memorize and Generate: Towards Faithful Table-to-Text Generation in Few Shots
This repo provides the source code & data of our paper: "Attend, Memorize and Generate: Towards Faithful Table-to-TextGeneration in Few Shots" https://aclanthology.org/2021.findings-emnlp.347 (EMNLP Findings 2021)

```
@inproceedings{zhao-etal-2021-attend-memorize,
    title = "Attend, Memorize and Generate: Towards Faithful Table-to-Text Generation in Few Shots",
    author = "Zhao, Wenting  and
      Liu, Ye  and
      Wan, Yao  and
      Yu, Philip",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.347",
    doi = "10.18653/v1/2021.findings-emnlp.347",
    pages = "4106--4117"
}

```

## Description
1. In AMG paper, we first do task-adaptive training for the AMG model using masked language model training objective to learn better model weights, then apply the checkpoint at second step to fine-tune the wiki-human/books/songs data on the model.

2. This repository provides the checkpoint after the task-adpative training, as well as data, code for the second fine-tuning step.

Feel free to reach out if you have any question!

## Installation
### install pytorch
```
  conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```
All the conda environment packages(including version) are listed in requirements_version.txt.

## Data
Data is from the Few-shot NLG with three domains : Humans, Songs and Books.

The original data, processed data, CKPT can be download from [AMG-DATA-MODEL-CKPT](https://drive.google.com/drive/folders/1-EHaDP3L2BYQTO_mekCr9cfkZz1p2-3F?usp=sharing) Google Drive.

### Data Organization:
```
AMG-DATA-MODEL-CKPT
├── 1-wiki-human-folder
│   ├── 1_original_data_500 (original data from wiki-dataset in Fewshot NLG)
│   └── 3_few_shot_plotmachine (the data format for the fine-tuning)
│   └── 4_pre_entity_embed_folder  (entity embedding for fine-tuning and inference)
│   └── CKPT (checkpoint after the task adaptive training)
├── 2-wiki-song-folder
│   ├── 1_original_data_500
│   └── 3_few_shot_plotmachine
│   └── 4_pre_entity_embed_folder
│   └── CKPT
├── 3-wiki-books-folder
│   ├── 1_original_data_500
│   └── 3_few_shot_plotmachine
│   └── 4_pre_entity_embed_folder
│   └── CKPT
```

Take wiki-Human folder as an example, 1_original_data_500 contains original data from wiki-dataset in Fewshot NLG, 3_few_shot_plotmachine has the data with format for the fine-tuning, while 4_pre_entity_embed_folder  provides entity embedding for fine-tuning and inference.

Unzip AMG-DATA-MODEL-CKPT, and place three folders: 1-wiki-human-folder,2-wiki-song-folder and 3-wiki-books-folder under path /data0 .

## Training
```
python ./biunilm/run_all_domains.py
```

## Inference
```
python ./biunilm/decode_all_domains.py
```


