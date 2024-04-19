import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

np.random.seed(42)


def ds_process_label(model_name="google-bert/bert-base-cased",title_weights=2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ## read the data and delete unnecessary columns
    full_df = pd.read_csv('MN-DS-news-classification.csv')
    full_df['data_id'] = range(1, full_df.shape[0] + 1)

    df = full_df.drop(['id','data_id','source','author','category_level_1','category_level_2','date','url','published','published_utc','collection_utc'], axis=1, inplace=False)

    ## encode the labels
    mlb = MultiLabelBinarizer()
    level1_encoded = pd.DataFrame(mlb.fit_transform(full_df['category_level_1'].apply(lambda x: [f'level1_{x}'])), columns=mlb.classes_)
    level2_encoded = pd.DataFrame(mlb.fit_transform(full_df['category_level_2'].apply(lambda x: [f'level2_{x}'])), columns=mlb.classes_)


    df_encoded = pd.concat([df, level1_encoded, level2_encoded], axis=1)

    ## split the data
    train_dev, test = train_test_split(df_encoded, test_size=0.2, random_state=42)
    train, dev = train_test_split(train_dev, test_size=0.125, random_state=42)


    train_dataset = Dataset.from_pandas(train)
    test_dataset = Dataset.from_pandas(test)
    dev_dataset = Dataset.from_pandas(dev)


    dataset_dict = DatasetDict({"train":train_dataset,"test":test_dataset,"dev":dev_dataset})

    labels = train.columns[2:].tolist()
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    label_names = df_encoded.columns[2:].tolist()

    tokenizer.model_max_length = 512
    
    # helper function to tokenize and feature the data
    def preprocess_data(examples):
        '''Preprocesses a batch of examples from a dataset and 
            returns a BatchEncoding object with the tokenized inputs'''

        content = [
                    (' ' + title) * title_weights + ' ' + text
                    for title, text in zip(examples['title'], examples['content'])
                ]
        tokenized = tokenizer(content,padding="max_length", truncation=True)
        labels = [examples[key] for key in label_names]
        tokenized["labels"] = np.array(list(zip(*labels)),dtype=float)
        return tokenized    


    ds = dataset_dict.map(preprocess_data, batched=True)

    # helper function to delete columns in dataset_dict
    def delete_columns(dataset_dict, columns_to_delete):
        modified_dataset_dict = DatasetDict()
        for split_name, split_dataset in dataset_dict.items():
            modified_split_dataset = split_dataset.remove_columns(columns_to_delete)
            modified_dataset_dict[split_name] = modified_split_dataset
        return modified_dataset_dict

    columns_to_delete = label_names

    ds = delete_columns(ds, columns_to_delete)

    # return the processed dataset, label names, id2label, label2id, and model_name
    return ds, label_names, id2label, label2id, tokenizer