import math
import warnings
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

# Ignore warnings
warnings.filterwarnings("ignore")

# get dataset of emotions
dataset = load_dataset('emotion')
print(dataset)

label_names = dataset["train"].features['label'].names
print(label_names)

# set format
dataset.set_format(type="pandas")
train_df = dataset['train'][:]
valid_df = dataset['validation'][:]
test_df = dataset['test'][:]
train_df.head()

train_df = train_df.groupby('label').apply(lambda x: x.sample(350)).reset_index(drop=True)
valid_df = valid_df.groupby('label').apply(lambda x: x.sample(70)).reset_index(drop=True)
test_df = test_df.groupby('label').apply(lambda x: x.sample(50)).reset_index(drop=True)

train_df['label'].value_counts()
valid_df['label'].value_counts()
test_df['label'].value_counts()

# Tokenization
PRETRAINED_LM = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM, do_lower_case=True)


# Return inputs_ids and attention_mask
def encode(docs):
    encoded_dict = tokenizer.batch_encode_plus(
        docs,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks


train_input_ids, train_att_masks = encode(train_df['text'].values.tolist())
valid_input_ids, valid_att_masks = encode(valid_df['text'].values.tolist())
test_input_ids, test_att_masks = encode(test_df['text'].values.tolist())

# DataSet
train_y = torch.LongTensor(train_df['label'].values.tolist())
valid_y = torch.LongTensor(valid_df['label'].values.tolist())
test_y = torch.LongTensor(test_df['label'].values.tolist())
train_y.size(), valid_y.size(), test_y.size()

BATCH_SIZE = 16
train_dataset = TensorDataset(train_input_ids, train_att_masks, train_y)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

valid_dataset = TensorDataset(valid_input_ids, valid_att_masks, valid_y)
valid_sampler = SequentialSampler(valid_dataset)
valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=BATCH_SIZE)

test_dataset = TensorDataset(test_input_ids, test_att_masks, test_y)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

# Classification
N_labels = len(train_df.label.unique())
model = BertForSequenceClassification.from_pretrained(
    PRETRAINED_LM,
    num_labels=N_labels,
    output_attentions=False,
    output_hidden_states=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#model = model.cuda()

# Fine Tuning
EPOCHS = 10  #30
LEARNING_RATE = 2e-6

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_dataloader) * EPOCHS)

# Training
train_loss_per_epoch = []
val_loss_per_epoch = []

for epoch_num in range(EPOCHS):
    print('Epoch: ', epoch_num + 1)

    # Training
    model.train()
    train_loss = 0
    for step_num, batch_data in enumerate(tqdm(train_dataloader, desc='Training')):
        input_ids, att_mask, labels = [data.to(device) for data in batch_data]
        output = model(input_ids=input_ids, attention_mask=att_mask, labels=labels)

        loss = output.loss
        train_loss += loss.item()

        model.zero_grad()
        loss.backward()
        del loss

        clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    train_loss_per_epoch.append(train_loss / (step_num + 1))

    # Validation
    model.eval()
    valid_loss = 0
    valid_pred = []
    with torch.no_grad():
        for step_num_e, batch_data in enumerate(tqdm(valid_dataloader, desc='Validation')):
            input_ids, att_mask, labels = [data.to(device) for data in batch_data]
            output = model(input_ids=input_ids, attention_mask=att_mask, labels=labels)

            loss = output.loss
            valid_loss += loss.item()

            valid_pred.append(np.argmax(output.logits.cpu().detach().numpy(), axis=-1))

    val_loss_per_epoch.append(valid_loss / (step_num_e + 1))
    valid_pred = np.concatenate(valid_pred)

    # Loss message
    print("{0}/{1} train loss: {2} ".format(step_num + 1, math.ceil(len(train_df) / BATCH_SIZE),
                                            train_loss / (step_num + 1)))
    print("{0}/{1} val loss: {2} ".format(step_num_e + 1, math.ceil(len(valid_df) / BATCH_SIZE),
                                          valid_loss / (step_num_e + 1)))

print('Classification')
print(classification_report(valid_pred, valid_df['label'].to_numpy(), target_names=label_names))

# Save model and tokenizer
model.save_pretrained('./model')
tokenizer.save_pretrained('./model/tokenizer')
