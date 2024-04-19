import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification,TextClassificationPipeline
import torch
import wandb
from transformers import EvalPrediction
from torch.nn import BCEWithLogitsLoss
from data_process import ds_process_label

torch.manual_seed(42)
np.random.seed(42)

model_name = "google-bert/bert-base-cased"

## add additional weight to the title, so that the model can learn to pay more attention to the title
title_weights = 2

# call data_process.py to get the dataset, labels, and model name
ds, label_names, id2label, label2id, tokenizer = ds_process_label(model_name=model_name,title_weights=title_weights)

# load the pretrained model using huggingface
model = AutoModelForSequenceClassification.from_pretrained(model_name,problem_type="multi_label_classification", id2label=id2label, label2id=label2id)

# set up the wandb environment
wandb.init(
    # set the wandb project where this run will be logged
    project="si630-proj-test",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
    "dataset": "normal",
    "epochs": 30,
    }
)

# set up the training arguments
multilabel_training_args = TrainingArguments(
    # FILL IN
    output_dir="./results/hierarchical",
    overwrite_output_dir=True,
    learning_rate=0.0001,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    do_eval=True,
    seed=12345,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    num_train_epochs=30,
    logging_dir="./logs/hierarchical",
    load_best_model_at_end=True,
    metric_for_best_model="eval_level2_f1",
    greater_is_better=True,
    save_total_limit=2,
    report_to="wandb",
)


# set up the evaluation function, notice we use different thresholds for level 1 and level 2, and calculate the f1 score for level 1 and level1+level2

def compute_hierarchical_metrics(eval_pred: EvalPrediction):
    # TODO:
    # 1. Get the logits and labels from the eval_pred
    # 2. Compute the probabilities from the logits using the sigmoid function
    # 3. Round the probabilities to get the binary predictions
    # 4. Calculate micro-averaged precision, recal, and F1
    # 5. Return the values as a dictionary with key names for indicating the metric
    logits, labels = eval_pred
    probabilities = 1 / (1 + np.exp(-logits))
    thresholds_level_1 = [0.5]*17
    thresholds_level_2 = [0.4]*109

    level1_out = (probabilities[:, :17] >= thresholds_level_1).astype(int)
    level2_out = (probabilities[:, 17:] >= thresholds_level_2).astype(int)
    outs = np.concatenate((level1_out, level2_out), axis=1)
    level2_f1 = f1_score(labels, outs, average='weighted')
    level1_f1 = f1_score(labels[:, :17], level1_out, average='weighted')

    return {'level1_f1':level1_f1, 'level2_f1': level2_f1, } 


# define a new Trainer class that takes in the class weights for the loss function, the weights are calculated based on the sparsity of the labels and the different levels of the labels

class WeightedTrainer(Trainer):
    def __init__(self, *args, level1_label_weight: float = 1.0, other_label_weight: float = 0.5, train_dataset, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataset = train_dataset
        num_labels = kwargs['model'].num_labels
        labels = torch.tensor(train_dataset['labels'],dtype=torch.float)
        pos_count = torch.sum(labels,dim=0)
        neg_count = labels.size(0)-pos_count
        
        # calculate the sparsity weights
        sparsity_weights = (neg_count/pos_count.clamp(min=1)).float()*2
        # calculate the level weights
        level_weights =torch.tensor([level1_label_weight] * 17 + [other_label_weight] * (num_labels - 17),dtype=torch.float)
        # combine the weights
        combined_weights = level_weights*sparsity_weights
        self.class_weights = torch.tensor(combined_weights).to(self.args.device)
        self.loss_fct = BCEWithLogitsLoss(pos_weight=self.class_weights)

    # use the combined weights to calculate the loss
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Flatten the logits and labels if they're not already.
        loss = self.loss_fct(logits.view(-1, model.num_labels), labels.float().view(-1, model.num_labels))

        return (loss, outputs) if return_outputs else loss


# set up the trainer
trainer = WeightedTrainer(
    # FILL IN
    model=model,
    args=multilabel_training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['dev'],
    tokenizer=tokenizer,
    compute_metrics=compute_hierarchical_metrics,
)   


trainer.train()


# evaluate the model on the test set of level 1+level 2 labels
test_preds = trainer.predict(ds['test'])
trainer.save_model("results/hierarchical_best")


logits = test_preds[0]
test_scores_array = 1 / (1 + np.exp(-logits))

test_labels = np.array(ds['test']['labels'])

thresholds_level_1 = [0.5]*17
thresholds_level_2 = [0.4]*109

pred_optimal = (test_scores_array >= thresholds_level_1+thresholds_level_2).astype(int)

f1 = f1_score(test_labels, pred_optimal, average='weighted')
precision = precision_score(test_labels, pred_optimal, average='weighted')
recall = recall_score(test_labels, pred_optimal, average='weighted')


print("==================================================")
print("F1 Score for 2 levels:", f1)
print("Precision Score for 2 levels:", precision)
print("Recall Score for 2 levels:", recall)
print("==================================================")
# -



# evaluate the model on the test set of level 1 labels
# +
from sklearn.metrics import f1_score
import numpy as np

test_sliced_scores= test_scores_array[:, :17]

test_labels = np.array(ds['test']['labels'])[:, :17]

thresholds_level_1 = [0.5]*17

pred_optimal = (test_sliced_scores >= thresholds_level_1).astype(int)

f1 = f1_score(test_labels, pred_optimal, average='weighted')
precision = precision_score(test_labels, pred_optimal, average='weighted')
recall = recall_score(test_labels, pred_optimal, average='weighted')

print("==================================================")
print("F1 Score for 1 levels:", f1)
print("Precision Score for 1 levels:", precision)
print("Recall Score for 1 levels:", recall)
print("==================================================")
# -


## calculate top3 accuracy for level 1 labels

df = ds["test"].to_pandas()
MODEL_NAME = "results/hierarchical_best"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer,device=0)

df['text'] = (' ' + df['title']) * 2 + ' ' + df['content']

tokenizer_kwargs = {'truncation':True,'max_length':512}

pred = [0]*df.shape[0]
for index, row in df.iterrows():
  true_labels = label_names[np.argmax(row['labels'])]
  pred_score = pipe(row['text'],top_k=None,**tokenizer_kwargs)
  top=0
  for i in pred_score:
    if top>=3:
      break
    if i['label'][:6] == 'level1':
      top+=1
      if i['label'] == true_labels:
        pred[index]=1
        break
      

print("==================================================")
print("Top3-accuracy:", sum(pred)/len(pred))
print("==================================================")