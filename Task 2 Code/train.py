from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
import json
import random
import torch
from tqdm.notebook import tqdm
from collections import Counter
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import default_data_collator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

# pretrained model name and datapath
model_name = "deepset/roberta-base-squad2"
datapath = "COVID-QA.json"
datapath_after = "covid_processed.json"

# cut context, leave a random number of character in front of the answer
def cut_context(start, length, context):
    # input: start,answer len, and context;
    # output new start and context

    # random number of char before answer
    num1 = random.randint(200,600)
    # after ans
    num2 = 1100 - num1
    # start and end of context cutting
    s = start - num1 if start - num1 >= 0 else 0
    e = start + length + num2 if start + length + num2 <= len(context) else len(context)
    # new answer start
    S = num1 if start - num1 >= 0 else start
    return S, context[s:e]

def reformat_data_into_squad(datapath,datapath_after):
    data = pd.read_json(datapath)
    covidqa = {}
    contents = []
    for i in tqdm(range(data.shape[0])):
        topic = data.iloc[i, 0]['paragraphs']
        for sub_para in topic:
            context = sub_para['context']
            title = sub_para['document_id']
            for q_a in sub_para['qas']:
                # cut context
                S, cut = cut_context(q_a['answers'][0]['answer_start'], len(q_a['answers'][0]['text']), context)
                content = {}
                content['context'] = cut
                content['title'] = title
                content['answers'] = {}
                content['answers']['answer_start'] = [S]
                content['answers']['text'] = q_a['answers'][0]['text']
                content['question'] = q_a['question']
                contents.append(content)
    covidqa['data'] = contents
    with open(datapath_after, 'w') as fp:
        json.dump(covidqa, fp, indent=4)

# preprocess data (tokenize the context and answers) -- customized according to the model
# refered to the huggingface website
def preprocess_function(examples):
    # for para in examples['paragraphs']:
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"])
        sequence_ids = inputs.sequence_ids(i)
        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# reformat data into squad format and load it
reformat_data_into_squad(datapath,datapath_after)
json_data = load_dataset('json',data_files = datapath_after, field = 'data')
# tokenize context and answer
tokenized_squad = json_data.map(preprocess_function,
                                batched=True,
                                remove_columns=json_data["train"].column_names)
# train_val_split
tokenized_squad = tokenized_squad['train'].train_test_split(test_size=0.3)
# data collator -- batch the data
data_collator = default_data_collator

# load pretrained model and tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# collaborate everything into training args
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps = 32
)
# define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
# fine tune it!
trainer.train()

# save the final model:
model_dir = "drive/MyDrive/TM Project/model"
tokenizer.save_pretrained(model_dir)
model.save_pretrained(model_dir)

#evaluation:
def GT_pred(trainer):
    #input: trainer
    #output: a list of [start,end] of ground truth
    # , and a list of [start,end] of prediction
    Predicted = list()
    GT = list()

    for batch in tqdm(trainer.get_eval_dataloader()):
        input_ids, attention_mask, label_start, label_end = batch.items()
        batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
        with torch.no_grad():
            output = trainer.model(**batch)
            loss, start_logits, end_logits = output.items()

            pred_start = np.argmax(start_logits[1].cpu().numpy(), axis=-1)
            pred_end = np.argmax(end_logits[1].cpu().numpy(), axis=-1)
            label_start = label_start[1].numpy()
            label_end = label_end[1].numpy()

            for i in range(len(pred_start)):
                Predicted.append([pred_start[i], pred_end[i]])
                GT.append([label_start[i], label_end[i]])
    return Predicted,GT

def intervalIntersection(A, B):
    # compute intersection of GT and Prediction
    ans = []
    i = j = 0
    while i < len(A) and j < len(B):
        lo = max(A[i][0], B[j][0])
        hi = min(A[i][1], B[j][1])
        if lo <= hi:
            ans.append([lo, hi])
        else:
            ans.append([0, 0])
        i += 1
        j += 1
    return ans


def compute_f1_em(pre, gt):
    # F1计算方法详见https://blog.csdn.net/z2536083458/article/details/96771806

    F1 = list()
    EM = list()

    allsame = intervalIntersection(pre, gt)
    for i, j, same in zip(pre, gt, allsame):
        len_pre = i[1] - i[0]
        len_gt = j[1] - j[0]
        len_same = same[1] - same[0]

        # compute f1
        if len_pre == 0 or len_gt == 0:
            f1 = 0
        elif len_same == 0:
            f1 = 0
        else:
            precision = 1.0 * len_same / (len_pre + 10 ** -16)
            recall = 1.0 * len_same / (len_gt + 10 ** -16)
            f1 = (2 * precision * recall) / (precision + recall)
        F1.append(f1)

        # compute EM
        if i == j and i[1] != 0 and j[1] != 0:
            EM.append(1)
        else:
            EM.append(0)
        # print('i:' ,i, 'j:', j, 'same:', same, 'f1:', f1, 'EM:', int(i == j), np.mean(F1), np.mean(EM))
    return F1, EM, np.mean(F1), np.mean(EM)

Predicted,GT = GT_pred(trainer)
print(GT[:10])
print(Predicted[:10])
F1, EM, F1_score, EM_score = compute_f1_em(Predicted, GT)
print(F1_score, EM_score)



