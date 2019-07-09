# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:32:17 2019

@author: Narshion Ngao
"""

import os
import json
from sklearn.model_selection import train_test_split
import torch
import logging
import numpy as np

CORPUS_PATH = os.path.join("C:/Users/user/UW/keshet - mwach_interaction/NarshionAnalysis/mWACH-NLP/scripts/bert_code/data", "messages.json")




data = json.load(open(CORPUS_PATH))
texts = [doc["text"] for (i, doc) in data.items()]
labels = [doc["label"] for (i, doc) in data.items()]

rest_texts, test_texts, rest_labels, test_labels = train_test_split(texts, labels, test_size=0.1, random_state=1)
train_texts, dev_texts, train_labels, dev_labels = train_test_split(rest_texts, rest_labels, test_size=0.1, random_state=1)

print("Train size:", len(train_texts))
print("Dev size:", len(dev_texts))
print("Test size:", len(test_texts))

label2idx = ['1','0']
# print(label2idx)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#initialize model
BERT_MODEL = "bert-base-uncased"
from pytorch_pretrained_bert.tokenization import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)


from pytorch_pretrained_bert import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = len(label2idx))
model.to(device)


#Preparing the data
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_SEQ_LENGTH=100

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        

def convert_examples_to_features(example_texts, example_labels, label2idx, max_seq_length, tokenizer, verbose=0):
    """Loads a data file into a list of `InputBatch`s."""
    
    features = []
    examples = zip(example_texts, example_labels)
    for (ex_index, (text, label)) in enumerate(examples):
        tokens = tokenizer.tokenize(text)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
            
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
            
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label2idx[label]
        if verbose and ex_index == 0:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label:" + str(label) + " id: " + str(label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

train_features = convert_examples_to_features(train_texts, train_labels, label2idx, MAX_SEQ_LENGTH, tokenizer, verbose=0)
dev_features = convert_examples_to_features(dev_texts, dev_labels, label2idx, MAX_SEQ_LENGTH, tokenizer)
test_features = convert_examples_to_features(test_texts, test_labels, label2idx, MAX_SEQ_LENGTH, tokenizer)


#we're going to initialize a data loader for our training, development and testing data. 
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

def get_data_loader(features, max_seq_length, batch_size): 

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([int(f.label_id) for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

BATCH_SIZE = 8

train_dataloader = get_data_loader(train_features, MAX_SEQ_LENGTH, BATCH_SIZE)
dev_dataloader = get_data_loader(dev_features, MAX_SEQ_LENGTH, BATCH_SIZE)
test_dataloader = get_data_loader(test_features, MAX_SEQ_LENGTH, BATCH_SIZE)

#Evaluation method
def evaluate(model, dataloader):

    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    for step, batch in enumerate(tqdm(dataloader, desc="Evaluation iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        outputs = np.argmax(logits, axis=1)
        label_ids = label_ids.to('cpu').numpy()
        
        predicted_labels += list(outputs)
        correct_labels += list(label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
        
    return eval_loss, correct_labels, predicted_labels

#Training
from pytorch_pretrained_bert.optimization import BertAdam

GRADIENT_ACCUMULATION_STEPS = 1
NUM_TRAIN_EPOCHS = 5
LEARNING_RATE = 5e-5
WARMUP_PROPORTION = 0.1

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

num_train_steps = int(len(train_texts) / BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_EPOCHS)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
t_total = num_train_steps

optimizer = BertAdam(optimizer_grouped_parameters,
                     LEARNING_RATE,
                     warmup=WARMUP_PROPORTION,
                     t_total=t_total)

#beginning actual training
import torch
import os
from tqdm import trange
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support

OUTPUT_DIR = "C:/Users/user/UW/keshet - mwach_interaction/NarshionAnalysis/mWACH-NLP/scripts/bert_code/outputs/"
MODEL_FILE_NAME = "pytorch_model.bin"
PATIENCE = 2

global_step = 0
model.train()
loss_history = []
for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Training iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        loss = model(input_ids, segment_ids, input_mask, label_ids)

        if GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            lr_this_step = LEARNING_RATE * warmup_linear(global_step/t_total, WARMUP_PROPORTION)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

    dev_loss, _, _ = evaluate(model, dev_dataloader)
    
    print("Loss history:", loss_history)
    print("Dev loss:", dev_loss)
    
    if len(loss_history) == 0 or dev_loss < min(loss_history):
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
    
    if len(loss_history) > 0 and dev_loss > max(loss_history[-PATIENCE:]): 
        print("No improvement on development set. Finish training.")
        break
        
    
    loss_history.append(dev_loss)
    
#Actual evaluation
BERT_MODEL = "bert-base-uncased"

#with open(TARGET_NAME_PATH) as i:
#    target_names = json.load(i)
    
target_names = label2idx

model_state_dict = torch.load(output_model_file)
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, state_dict=model_state_dict, num_labels = len(target_names))
model.to(device)

model.eval()

_, train_correct, train_predicted = evaluate(model, train_dataloader)
_, dev_correct, dev_predicted = evaluate(model, dev_dataloader)
_, test_correct, test_predicted = evaluate(model, test_dataloader)

print("Training performance:", precision_recall_fscore_support(train_correct, train_predicted, average="micro"))
print("Development performance:", precision_recall_fscore_support(dev_correct, dev_predicted, average="micro"))
print("Test performance:", precision_recall_fscore_support(test_correct, test_predicted, average="micro"))

print(classification_report(test_correct, test_predicted, target_names=target_names))










