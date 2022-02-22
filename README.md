# AMG
Data, Checkpoint, and Model for EMNLP Findings paper "Attend, Memorize and Generate: Towards Faithful Table-to-TextGeneration in Few Shots" https://aclanthology.org/2021.findings-emnlp.347/

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
│   ├── begin-with-the-crazy-ideas.textile
│   └── on-simplicity-in-technology.markdown
│   └── on-simplicity-in-technology.markdown
│   └── on-simplicity-in-technology.markdown
├── 2-wiki-song-folder
│   ├── footer.html
│   └── header.html
├── 3-wiki-books-folder
    ├── default.html
    └── post.html
```
### Step1:
Unzip AMG-DATA-MODEL-CKPT, and place three folders: 1-wiki-human-folder,2-wiki-song-folder and 3-wiki-books-folder under path /data0 .
