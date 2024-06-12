import torch
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer,AutoModel
import json, os
from concept_helper import submodular_select

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 使用分词器对文本进行分词和编码
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
@torch.no_grad()
def get_concepts(task_name, num_samples_per_class):
    
    file_p = os.path.join("dataset",task_name,"train","train.tsv")
    samples = []
    with open(file_p, "r") as f:
        data = f.readlines()
    data = data[1:]
    sample_cnt = {}
    for line in data:
        line = line.strip()
        sentence, label = line.split("\t")
        if label not in sample_cnt.keys():
            sample_cnt.update({label : 1})
            samples.append((sentence, label))
        elif sample_cnt[label] < num_samples_per_class:
            sample_cnt[label] += 1
            samples.append((sentence, label))
        else:
            break
    sentences = [sample[0] for sample in samples]
    labels = [int(sample[1]) for sample in samples]
    model = AutoModel.from_pretrained("/root/autodl-tmp/bert-large-uncased")
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/bert-large-uncased')

    with open("./submodular/labels.json","r") as f:
        label2concept = json.load(f)
    if task_name == "sst2":
        concept_list = label2concept["postive"] + label2concept["negative"]

    batch = tokenizer(sentences, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
    outputs = model(input_ids = batch['input_ids'], attention_mask = batch["attention_mask"])
    hidden_states = outputs.last_hidden_state[:,0,:]
    
    encoding = tokenizer(concept_list, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
    
    concepts_states = model(input_ids = encoding['input_ids'], attention_mask = encoding['attention_mask'])  
    concept_feat = concepts_states.last_hidden_state[:,0,:]
    
    if task_name == "sst2":
        concept2cls = torch.LongTensor([1] * 50 + [0] * 50)
    num_concepts = 10
    num_images_per_class = 16
    submodular_weights = [1e7, 1]
    select_idx =  submodular_select(text_feat = hidden_states, concept_feat = concept_feat,  concept2cls = concept2cls, num_concepts = num_concepts, num_images_per_class = [16,16], submodular_weights = submodular_weights)
    print(select_idx)

get_concepts("sst2", 16)

        
        
    
        
