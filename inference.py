import torch
from transformers import BertTokenizer, BertForTokenClassification  # Change here

model_path = 'model_NER'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForTokenClassification.from_pretrained(model_path)

model.eval()
id2tag = {0: 'O', 1: 'B-LOG', 2: 'I-LOG'}

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    predictions = outputs.logits.argmax(dim=2).squeeze().tolist()
    predicted_labels = [id2tag[pred] for pred in predictions]

    for i in range(1, len(tokens)):
        if tokens[i].startswith('##') and predicted_labels[i-1] in ['B-LOG', 'I-LOG']:
            predicted_labels[i] = 'I-LOG'

    prediction_pairs = list(zip(tokens, predicted_labels))
    return prediction_pairs

if __name__ == "__main__":
    text ='K2, known for its challenging climbs and technical difficulties, is often considered a more daunting ascent than Everest, despite Everest being the worlds highest peak.'
    prediction = predict(text)

    print(prediction)