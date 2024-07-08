from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

app = Flask(__name__)

model = BertForSequenceClassification.from_pretrained('./model')
tokenizer = BertTokenizer.from_pretrained('./model/tokenizer')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Predict
        inputs_request = [str(x) for x in request.form.values()]
        print(f'request:', inputs_request)
        predictions = predict_new_texts(inputs_request)

        # Load names for labels
        #label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        label_names = ['tristeza', 'alegría', 'amor', 'ira', 'miedo', 'sorpresa']

        predicted_labels = [label_names[pred] for pred in predictions]
        return render_template('index.html', prediction_text=f'Usted siente: {predicted_labels[0]}')


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

def predict_new_texts(new_texts):
    input_ids, attention_masks = encode(new_texts)
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=16)

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in prediction_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_masks = batch

            outputs = model(input_ids=input_ids, attention_mask=attention_masks)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).flatten()

            predictions.extend(preds.cpu().numpy())

    return predictions


if __name__ == '__main__':
    app.run(debug=True)
