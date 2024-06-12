import torch
import torch.nn as nn
from typing import List, Dict, Optional


def generate_default_inputs(model, model_type:str ,  batch: Dict[str, torch.Tensor], prompt_encoder_type:str = "lstm") -> Dict[str, torch.Tensor]:


    input_ids = batch['input_ids']
    bz = batch['input_ids'].shape[0]
    block_flag = batch["block_flag"]
    # print(input_ids.size(), block_flag.size())
    
    

    if model_type == "albert":
        raw_embeds = model.model.albert.embeddings.word_embeddings(input_ids)
    elif model_type == "bert":
        raw_embeds = model.model.bert.embeddings.word_embeddings(input_ids)
    elif model_type == "roberta":
        raw_embeds = model.model.roberta.embeddings.word_embeddings(input_ids)
    elif model_type == "GPT2":
        raw_embeds = model.model.transformer.wte(input_ids)
    
    if "concept" not in batch.keys():
        replace_embeds = model.prompt_embeddings(
            torch.LongTensor(list(range(model.prompt_length))).cuda()) # [1,1024]
        replace_embeds = replace_embeds.unsqueeze(0) # [batch_size, prompt_length, embed_size]
    else:
        select_concept = batch["concept"] # num_concepts,emb_size


        replace_embeds = model.score @ select_concept  # 1, emb_size
        replace_embeds = replace_embeds.unsqueeze(0) # 1, prompt_length, embed_size
        
    if prompt_encoder_type == "lstm":
        replace_embeds = model.lstm_head(replace_embeds)[0]  # [batch_size, seq_len, 2 * hidden_dim]
        if model.prompt_length == 1:
            replace_embeds = model.mlp_head(replace_embeds)
        else:
            replace_embeds = model.mlp_head(replace_embeds).squeeze()

    elif prompt_encoder_type == "mlp":
        replace_embeds = model.mlp(replace_embeds)
    else:
        raise ValueError("unknown prompt_encoder_type.")
    blocked_indices = (block_flag == 1).nonzero().reshape((bz, model.prompt_length, 2))[:, :, 1]

    for bidx in range(bz):
        for i in range(blocked_indices.shape[1]):
            raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]

    inputs = {'inputs_embeds': raw_embeds, 'attention_mask': batch['attention_mask']}

    if model_type in ['bert']:
        inputs['token_type_ids'] = batch['token_type_ids']

    return inputs
    
    



def integrated_gradients(model, batch, encoding):
    '''
    
    '''
    # print(y)
    model_type = model.config.model_type
    ctp = model.model
    for name, weight in ctp.named_parameters():
        # print(name)
        if model_type == "bert":
            if 'embedding' not in name:
                weight.requires_grad = False
        elif model_type == "GPT2":
            if "wte" not in name:
                weight.requires_grad = False

    # init_embed_weight = model.word_attn.embedding.weight.data
    if model_type == "bert":
        init_embed_weight = ctp.model.bert.embeddings.word_embeddings.weight.data
    
    elif model_type == "GPT2":
        init_embed_weight = ctp.model.transformer.wte.weight.data
    
    # x = text_token # B,N

    # init_word_embedding = init_embed_weight[x] # B,N,D
    # print(init_word_embedding.size())
    inputs = generate_default_inputs(ctp, model_type = model_type, batch = batch)
    # init_word_embedding = inputs["inputs_embeds"]
    # 获取baseline
    baseline = 0 * init_embed_weight
    

    steps = 20
    gradient_list = []
    # print(encoding["input_ids"].shape) # 100,256
    for i in range(steps + 1):
        scale_weight = baseline + float(i / steps) * (init_embed_weight - baseline)

        if model_type == "bert":
            # model.bertmodel.embeddings.word_embeddings.weight.data = scale_weight
            ctp.model.bert.embeddings.word_embeddings.weight.data = scale_weight
        elif model_type == "GPT2":
            ctp.model.transformer.wte.weight.data = scale_weight
        # generate word embeddings 
        input_embeds = generate_default_inputs(ctp, model_type = model_type, batch = batch)
            

        # pred = model(input_ids=x, attention_mask=token_mask) # B,C
        pred = model.mlm_eval_step2(batch, inputs = input_embeds["inputs_embeds"])
        
        pred = pred.view(-1, len(model.config.label_list))
        
        # print(pred)
        target_pred = pred[:, batch["labels"]] # B
        # print("target_pred:{}".format(target_pred.requires_grad))

        # print(target_pred[0].requires_grad)
        # if target_pred[0].requires_grad is False:
        # target_pred.requires_grad = True
        
        target_pred.sum().backward() 
        
        # print(model.bertmodel.embeddings.word_embeddings.weight.grad_fn)
        if model_type == "bert":
            grads = ctp.model.bert.embeddings.word_embeddings.weight.grad[encoding["input_ids"]].unsqueeze(1)
        
        elif model_type == "GPT2":
            grads = ctp.model.transformer.wte.weight.grad[encoding["input_ids"]].unsqueeze(1)
        # print(grads.shape) 100, 1, 256, 1024
        # grads = grads
        gradient_list.append(grads)
        # print(gradient_list[-1])
        ctp.zero_grad()

    # steps,input_len,dim
    gradient_list = torch.cat(gradient_list,dim=1)# B,S,L,D
    # gradient_list = torch.from_numpy(np.asarray(gradient_list)) 
    
    # input_len,dim
    avg_gradient = torch.mean(gradient_list,dim=1)# B,L,D 
    # avg_gradient = avg_gradient.detach().cpu().numpy()
    # x-baseline

    # delta_x = init_word_embedding 
    delta_x = ctp.encode(input_ids = encoding["input_ids"], attention_mask = encoding["attention_mask"], return_all=True)
    # 100,256,1024
    # delta_x = delta_x.detach().cpu().numpy()
    # print(delta_x.shape)

    # print(avg_gradient.shape, delta_x.shape)
    # torch.Size([100, 256, 1024]) torch.Size([128, 256, 1024])

    ig = avg_gradient * delta_x # 100,256,1024

    # word_ig = np.sum(ig, axis=1)
    word_ig = torch.sum(ig,dim=-1) # B,L 100,256
    ctp.zero_grad()
    
    concept_ig = torch.sum(word_ig, dim = -1) # B
    
    return concept_ig


def gradients(wrapper, batch, encoding):
    '''
    
    '''
    # print(y)
    # 除embedding层外，固定住所有的模型参数
    model_type = wrapper.config.model_type
    ctp = wrapper.model
    for name, weight in ctp.named_parameters():
        if model_type == "bert":
            if 'embedding' not in name:
                weight.requires_grad = False
        elif model_type == "GPT2":
            if "wte" not in name:
                weight.requires_grad = False

    
    
    
    inputs = generate_default_inputs(ctp, model_type = model_type, batch = batch)
    init_word_embedding = inputs["inputs_embeds"]
   
    # gradient_list = []

   
        
    pred = wrapper.mlm_eval_step2(batch, init_word_embedding)
        
    pred = pred.view(-1, len(wrapper.config.label_list))
        
        
    target_pred = pred[:, batch["labels"]] # B

    target_pred.sum().backward() 
        
    # print(model.bertmodel.embeddings.word_embeddings.weight.grad_fn)
    if model_type == "bert":
        grads = ctp.model.bert.embeddings.word_embeddings.weight.grad[encoding["input_ids"]].unsqueeze(1)
    
    elif model_type == "GPT2":
        grads = ctp.model.transformer.wte.weight.grad[encoding["input_ids"]].unsqueeze(1)
        # print(grads.shape)
        # grads = grads
    # gradient_list.append(grads)
        # print(gradient_list[-1])
    ctp.zero_grad()
    word_ig = torch.sum(grads,dim=-1) # B,L
    concept_ig = torch.sum(word_ig, dim = -1) # B
    ctp.zero_grad()
    
    return concept_ig

