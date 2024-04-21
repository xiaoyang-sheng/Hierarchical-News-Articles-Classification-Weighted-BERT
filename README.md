# Hierarchical-News-Articles-Classification-Weighted-BERT
Project of Umich SI 630: Natural Language Processing: Algorithms and People

This project is aimed to classify the topic of news article into 2 levels, with 17 level-1 labels and 109 level-2 labels.

You may refer to **report.pdf** for details.

## Installation
Package used including pandas, numpy, dataset, transformers, wandb, torch and sklearn.

requirements.txt is just a reference for the environment. 
```shell
$ pip install -r requirements.txt
```

## Dataset Setup

-[MN-DS: A Multilabeled News Dataset for News Articles Hierarchical Classification](https://zenodo.org/records/7394851)

## Quickstart

```shell
python weighted_bert.py
```

## Experiment Logs

Test metrics on level 1:
- Top 3 Accuracy for level 1: 95.6%
- F1 Score for level 1: 0.867
- Precision Score for level 1: 84.4%
- Recall Score for level 1: 89.2%

Test metrics on 2 levels:
- F1 Score for 2 levels: 0.793
- Precision Score for 2 levels: 74.7%
- Recall Score for 2 levels: 86.3%
