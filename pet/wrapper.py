# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains code for wrapping a transformer language model and
provides convenience methods for training and inference.
"""
import json
import jsonpickle
import os
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import trange, tqdm
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, \
    BertForMaskedLM, RobertaForMaskedLM, BertConfig, BertTokenizer, RobertaConfig, \
    RobertaTokenizer, AlbertForMaskedLM, AlbertTokenizer, AlbertConfig, \
    GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification,BertForSequenceClassification
from transformers.data.metrics import simple_accuracy
from torch.nn.functional import softmax
import logging as log
from pet import preprocessor
from data_utils.task_processors import TASK_HELPERS
from pet.config import WrapperConfig, EvalConfig
from pet.utils import InputFeatures, DictDataset, distillation_loss, exact_match
from submodular.concept_helper import submodular_select
import random
from visualize import compute_kernel_bias
from utils.AOPC import integrated_gradients, gradients
import numpy as np
logger = log.getLogger('root')

CONFIG_NAME = 'wrapper_config.json'
MLM_WRAPPER = "mlm"

WRAPPER_TYPES = [MLM_WRAPPER]

PREPROCESSORS = {
    MLM_WRAPPER: preprocessor.MLMPreprocessor,
}

MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        MLM_WRAPPER: BertForMaskedLM,
        'cls': BertForSequenceClassification
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        MLM_WRAPPER: RobertaForMaskedLM
    },
    'albert': {
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        MLM_WRAPPER: AlbertForMaskedLM
    },
    'GPT2': {
        'config': GPT2Config,
        'tokenizer': GPT2Tokenizer, 
        MLM_WRAPPER: GPT2LMHeadModel
    }
}

EVALUATION_STEP_FUNCTIONS = {
    MLM_WRAPPER: lambda wrapper: wrapper.mlm_eval_step
}

TRAIN_STEP_FUNCTIONS = {
    MLM_WRAPPER: lambda wrapper: wrapper.mlm_train_step
}


class BOWRegression(nn.Module):
    def __init__(self,prompt_length, concept_size):
        super().__init__()
        
        self.weight = nn.Linear(concept_size, prompt_length) # N_c,N_q
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs_embeds=None):
        # N_c,d
        return self.sigmoid(self.weight(inputs_embeds.T)).T

class PLMSequenceClassifier(torch.nn.Module):
    def __init__(self, config:WrapperConfig, tokenizer) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        config_class = MODEL_CLASSES[self.config.model_type]['config']
        model_config = config_class.from_pretrained(
            config.model_name_or_path,
            num_labels=len(config.label_list),
            finetuning_task=config.task_name,
            cache_dir=config.cache_dir if config.cache_dir else None,
            use_cache=False)
        model_class = MODEL_CLASSES[self.config.model_type]["cls"]
        self.model = model_class.from_pretrained(
            config.model_name_or_path,
            config=model_config,
            cache_dir=config.cache_dir if config.cache_dir else None, ignore_mismatched_sizes=True)
    def forward(self, input_ids, attention_mask,labels):
        loss,logits = self.model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)[:2]
        return loss,logits



class ContinuousPrompt(torch.nn.Module):
    def __init__(self, config:WrapperConfig, tokenizer, vocab1500 = None, args = None):
        super(ContinuousPrompt, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.embed_size = config.embed_size
        self.hidden_size = self.embed_size
        if self.config.pattern_id <= 2:
            self.prompt_length = self.config.pattern_id # The pattern_id is supposed to indicate the number of continuous prompt tokens.
        else:
            # 3 - 1, 4 -2 , 5 - 1, 6 -2 
            
            if config.pattern_id == 3:
                self.prompt_length = 1
            elif config.pattern_id == 4:
                self.prompt_length = 2
            elif config.pattern_id == 5:
                self.prompt_length = 1
            elif config.pattern_id == 6:
                self.prompt_length = 2

        config_class = MODEL_CLASSES[self.config.model_type]['config']
        model_config = config_class.from_pretrained(
            config.model_name_or_path,
            num_labels=len(config.label_list),
            finetuning_task=config.task_name,
            cache_dir=config.cache_dir if config.cache_dir else None,
            use_cache=False)


        model_class = MODEL_CLASSES[self.config.model_type][MLM_WRAPPER]
        self.model = model_class.from_pretrained(
            config.model_name_or_path,
            config=model_config,
            cache_dir=config.cache_dir if config.cache_dir else None)


        self.prompt_embeddings = torch.nn.Embedding(self.prompt_length, self.embed_size)
        if config.prompt_encoder_type == "lstm":
            self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                           hidden_size=self.hidden_size,
                                           num_layers=2,
                                           bidirectional=True,
                                           batch_first=True)
            self.mlp_head = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_size, self.hidden_size))
        elif config.prompt_encoder_type == "mlp":
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_size, self.hidden_size))
        else:
            raise ValueError('unknown prompt_encoder_type.')
        if vocab1500 is not None:
            # self.r = torch.empty((self.prompt_length, 1500)).uniform_(0, 1 / config.num_concepts).cuda()
            self.r = torch.empty((self.prompt_length, len(vocab1500))).normal_(0, 1).cuda()
        if config.submodular :
            self.score = torch.empty((self.prompt_length, config.num_concepts)).uniform_(0, 1 / config.num_concepts).cuda()
        if args is not None and args.train_bow != -1:
            self.bow = BOWRegression(prompt_length = args.prompt_length, concept_size=args.num_concepts)
    @torch.no_grad()
    def encode(self, input_ids = None, attention_mask = None, return_all = False):
        if self.config.model_type == "bert":
            if self.config.pool == "CLS":
   
                if not return_all:
                    return self.model.bert(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state[:,0,:]
                else:
                    return self.model.bert(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state
            elif self.config.pool == "mean":
                return self.model.bert(input_ids = input_ids, attention_mask = attention_mask).mean(1)
        elif self.config.model_type == "GPT2":
            if not return_all:
                
                embeddings = self.model.transformer(input_ids = input_ids, attention_mask = attention_mask)[0][:,-1,:]
            else:
                embeddings = self.model.transformer(input_ids = input_ids, attention_mask = attention_mask)[0]
            return embeddings
            
        
    def forward(self, inputs_embeds=None, attention_mask=None, token_type_ids=None, labels=None):
        return self.model(inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        labels=labels,
                        token_type_ids=token_type_ids)
        





class TransformerModelWrapper:
    """A wrapper around a Transformer-based language model."""

    def __init__(self, config: WrapperConfig, not_pt = None, vocab1500 = None, all_args = None):
        self.config = config
        self.submodular = config.submodular
        self.vocab1500 = vocab1500
        tokenizer_class = MODEL_CLASSES[self.config.model_type]['tokenizer']
        self.tokenizer = tokenizer_class.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir if config.cache_dir else None)
        self.args = all_args
       
        if not_pt is None:
            self.model = ContinuousPrompt(config, self.tokenizer, vocab1500 = vocab1500, args = all_args)
        else:
            self.model = PLMSequenceClassifier(config, self.tokenizer)
        
        if self.config.model_type == "GPT2":
            self.transformer = self.model.model.transformer
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['[mask]'], 'pad_token':'[pad]'})
            self.transformer.resize_token_embeddings(len(self.tokenizer))
            
            with torch.no_grad():
                mean_pooling = torch.mean(self.transformer.wte.weight.data[:-2],dim = 0)
            guassian_noise = torch.randn_like(self.transformer.wte.weight.data[-1]) 

            self.transformer.wte.weight.data[-2].copy_(mean_pooling.data)
            self.transformer.wte.weight.data[-1].copy_(self.transformer.wte.weight.data[-3])
            self.transformer.wte.weight.data[-1].add_(guassian_noise.data)
            
        self.preprocessor = PREPROCESSORS[MLM_WRAPPER](self,
                                                       self.config.task_name,
                                                       self.config.pattern_id)

        self.task_helper = TASK_HELPERS[self.config.task_name](self) if self.config.task_name in TASK_HELPERS else None


        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.cuda()


    def save(self, path: str) -> None:
        logger.info("Saving models.")
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        if self.args.train_bow == -1:
            model_to_save.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            self._save_config(path)

            if self.config.prompt_encoder_type == "lstm":
                state = {
                    "prompt_embeddings": model_to_save.prompt_embeddings.state_dict(),
                    "lstm_head": model_to_save.lstm_head.state_dict(),
                    "mlp_head": model_to_save.mlp_head.state_dict()
                }
            elif self.config.prompt_encoder_type == "mlp":
                state = {
                    "prompt_embeddings": model_to_save.prompt_embeddings.state_dict(),
                    "mlp": model_to_save.mlp.state_dict()
                }
            else:
                raise ValueError("unknown prompt_encoder_type.")

            save_path_file = os.path.join(path, "embeddings.pth")
            torch.save(state, save_path_file)
        else:
            # torch.save(model_to_save, path)
            # model_to_save.save_pretrained(path)
            # torch.save(model_to_save.bow.state_dict(), path)
            # model_to_save.bow.save(path)
            weight = model_to_save.bow.weight.weight.data.detach().cpu().numpy()
            np.save(os.path.join(path, "weight.npy"), weight)
            # with open(os.path.join(path, "weight.npy"),"rb")


    @classmethod
    def from_pretrained(cls, path: str) -> 'TransformerModelWrapper':
        """Load a pretrained wrapper from a given path."""

        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)

        tokenizer_class = MODEL_CLASSES[wrapper.config.model_type]['tokenizer']
        wrapper.tokenizer = tokenizer_class.from_pretrained(path)
        
        wrapper.model = ContinuousPrompt(wrapper.config, wrapper.tokenizer)
        model_class = MODEL_CLASSES[wrapper.config.model_type][MLM_WRAPPER]

        wrapper.model.model = model_class.from_pretrained(path, ignore_mismatched_sizes=True)


        save_path_file = os.path.join(path, "embeddings.pth")
        data = torch.load(save_path_file)
        wrapper.model.prompt_embeddings.load_state_dict(data["prompt_embeddings"])
        if "lstm_head" in data:
            assert ("mlp_head" in data)
            wrapper.model.lstm_head.load_state_dict(data["lstm_head"])
            wrapper.model.mlp_head.load_state_dict(data["mlp_head"])
        if "mlp" in data:
            wrapper.model.mlp_head.load_state_dict(data["mlp"])

        wrapper.preprocessor = PREPROCESSORS[MLM_WRAPPER](wrapper, wrapper.config.task_name, wrapper.config.pattern_id)

        wrapper.task_helper = TASK_HELPERS[wrapper.config.task_name](wrapper) \
            if wrapper.config.task_name in TASK_HELPERS else None

        if torch.cuda.device_count() > 1:
            wrapper.model = torch.nn.DataParallel(wrapper.model)
        wrapper.model.cuda()

        return wrapper


    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), 'w') as f:
            f.write(jsonpickle.encode(self.config))


    @staticmethod
    def _load_config(path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), 'r') as f:
            return jsonpickle.decode(f.read())
    
    
    
    def train(self,
              train_data:List[InputExample],
              eval_data:List[InputExample],
              dev32_data:List[InputExample],
              eval_config:EvalConfig,
              pattern_iter_output_dir,
              per_gpu_train_batch_size: int = 8,
              n_gpu: int = 1,
              num_train_epochs: int = 3,
              gradient_accumulation_steps: int = 1,
              weight_decay: float = 0.0,
              learning_rate: float = 5e-5,
              adam_epsilon: float = 1e-8,
              warmup_steps=0,
              max_grad_norm: float = 1,
              logging_steps: int = 50,
              max_steps=-1, 
              label_set=None, 
              concept2cls = None, 
              num_concepts = -1,
              num_images_per_class = -1,
              submodular_weights = None,
              r = None, not_pt = None, vocab1500 = None, word_lr = None,
              **_):
        """
        Train the underlying language model.

        :param train_data: the training examples to use
        :param per_gpu_train_batch_size: the number of training examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train
        :param gradient_accumulation_steps: the number of gradient accumulation steps before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the learning rate to use
        :param adam_epsilon: epsilon parameter for the Adam optimizer
        :param warmup_steps: the number of warmup steps
        :param max_grad_norm: the maximum norm for the gradient
        :param logging_steps: the number of steps after which logging information is printed
        :param max_steps: the maximum number of training steps, overrides ``num_train_epochs``
        :return: a tuple consisting of the total number of steps and the average training loss
        """

        train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
        train_dataset = self._generate_dataset(train_data)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (max(1, len(train_dataloader) // gradient_accumulation_steps)) + 1
        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

        print("\n")
        print("num_steps_per_dataset:")
        print(len(train_dataloader) // gradient_accumulation_steps)
        print("total_steps:")
        print(t_total)
        print("num_train_epochs:")
        print(num_train_epochs)
        print("\n")


        cur_model = self.model.module if hasattr(self.model, 'module') else self.model

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in cur_model.model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay},
            {'params': [p for n, p in cur_model.model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        

        if self.args.train_bow == -1: 
            if not_pt is None: # ptuning & concept
                if self.config.prompt_encoder_type == "lstm":
                    embedding_parameters = [
                        {'params': [p for p in cur_model.lstm_head.parameters()]},
                        {'params': [p for p in cur_model.mlp_head.parameters()]},
                        {'params': [p for p in cur_model.prompt_embeddings.parameters()]}
                    ]
                elif self.config.prompt_encoder_type == "mlp":
                    embedding_parameters = [
                        {'params': [p for p in cur_model.mlp.parameters()]},
                        {'params': [p for p in cur_model.prompt_embeddings.parameters()]}
                    ]
                if  self.config.submodular:
                    if self.config.r is None:
                        embedding_parameters.append(
                            {'params': [p for p in cur_model.score]}
                        )
                if self.vocab1500 is not None:
                    r_embedding_parameters = [
                        {"params":[p for p in cur_model.r]} # 1,1500
                    ]
                    r_optimizer = AdamW(r_embedding_parameters, lr=word_lr, eps=adam_epsilon)
                    # vocab = self.tokenizer.vocab
                    # vocab1500 
                    tok_vocab = [self.tokenizer.encode(x, add_special_tokens=False)[0] for x in vocab1500]
                    embeds = cur_model.model.bert.embeddings.word_embeddings
                    vocab_embeds = embeds.weight.data[tok_vocab] # 1500,1024
                    # embedding_parameters.append(
                    #     {"params":[p for p in cur_model.r]}
                    # )
        
                embedding_optimizer = AdamW(embedding_parameters, lr=learning_rate, eps=adam_epsilon)
                embedding_scheduler = get_linear_schedule_with_warmup(embedding_optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            # training BOW 
            embedding_parameters = [
                        {'params': [p for p in cur_model.lstm_head.parameters()]},
                        {'params': [p for p in cur_model.mlp_head.parameters()]},
                        {'params': [p for p in cur_model.prompt_embeddings.parameters()]}
                    ]
            embedding_parameters.append({'params':[p for p in cur_model.bow.parameters()]})
            embedding_optimizer = AdamW(embedding_parameters, lr = learning_rate)
            embedding_scheduler = get_linear_schedule_with_warmup(embedding_optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)




        writer = SummaryWriter(log_dir=os.path.join(self.config.output_dir, "writer_logs"))

        ### TODO
        prev_loss = 0.0
        best_dev32_acc = 0.0
        best_dev32_f1 = 0.0
        best_global_step = 0
        best_loss = 0.0
        early_stop_epoch = 0

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        encoding = None
        if self.config.submodular:
            # print(f"label_set:{len(label_set)}")
            encoding = self.tokenizer(label_set, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
            encoding["input_ids"] = encoding["input_ids"].cuda()
            # print(f"encoding.shape:{encoding["input_ids"].shape}")
            encoding["attention_mask"] = encoding["attention_mask"].cuda()
            concept2cls = torch.LongTensor(concept2cls).cuda()
            

        train_iterator = trange(int(num_train_epochs), desc="Epoch")
        
        
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                
                batch = {k: t.cuda() for k, t in batch.items()}

                if self.config.submodular:
                    text_feat = self.model.encode(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"])
                    concept_feat = self.model.encode(input_ids = encoding["input_ids"], attention_mask = encoding["attention_mask"])
                    select_idx = None
                    if  r is None:
                        # print(num_concepts)
                        select_idx =  submodular_select(text_feat, concept_feat,  concept2cls, num_concepts, num_images_per_class, submodular_weights)

                        
                        select_concept_feat = concept_feat[select_idx] # num_concepts, D
                        batch["concept"] = select_concept_feat
                    else:
                        select_idx = sorted(random.sample(range(len(concept_feat)), num_concepts))
                        select_concept_feat = concept_feat[select_idx]
                        batch["concept"] = select_concept_feat 
                        
                if self.args.train_bow == -1:
                    if not_pt is None:
                        loss = self.task_helper.train_step(batch) if self.task_helper else None
                        if loss is None:
                            loss = TRAIN_STEP_FUNCTIONS[MLM_WRAPPER](self)(batch)
                    elif self.vocab1500 is None:
                        loss, _ = self.model(batch["input_ids"], batch["attention_mask"],batch["labels"])
                    else:
                        # word explanation
                        batch["vocab"] = vocab_embeds
                        loss = TRAIN_STEP_FUNCTIONS[MLM_WRAPPER](self)(batch)
                else:
                    # train a BOW classifier
                    batch["bow_input"] =  select_concept_feat
                    loss = TRAIN_STEP_FUNCTIONS[MLM_WRAPPER](self)(batch)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                if (step + 1) % gradient_accumulation_steps == 0:
                    ## TODO
                    writer.add_scalar("train_loss", (tr_loss - prev_loss), global_step=global_step)
                    prev_loss = tr_loss

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    if self.args.train_bow != -1:
                        embedding_optimizer.step()
                        embedding_scheduler.step()
                    elif not_pt is None:
                        embedding_optimizer.step()
                        embedding_scheduler.step()
                    

                    self.model.zero_grad()
                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs['learning_rate'] = learning_rate_scalar
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss
                        print(json.dumps({**logs, **{'step': global_step}}))

                    ## TODO
                    if global_step % self.config.eval_every_step == -1:
                        dev32_scores = self.eval_dev(dev32_data, eval_config, n_gpu, 
                                                     encoding = encoding, 
                                                     concept2cls = concept2cls, 
                                                     num_concepts = num_concepts,
                                                     num_images_per_class = num_images_per_class,
                                                     submodular_weights = submodular_weights, r=r,not_pt=not_pt,vocab_embeds = vocab_embeds)

                        if self.config.task_name in ["cb", "record", "multirc"]:
                            f1_str = "f1" if self.config.task_name != "cb" else "f1-macro"
                            if dev32_scores["acc"] >= best_dev32_acc and dev32_scores[f1_str] >= best_dev32_f1:

                                if dev32_scores["acc"] > best_dev32_acc and dev32_scores[f1_str] > best_dev32_f1:
                                    early_stop_epoch = 0
                                else:
                                    early_stop_epoch += 1

                                best_dev32_acc = dev32_scores["acc"]
                                best_dev32_f1 = dev32_scores[f1_str]
                                best_global_step = global_step
                                best_loss = tr_loss

                                logger.info("Saving trained model at {}...".format(pattern_iter_output_dir))
                                logger.info("best_dev32_acc: %.4f | best_dev32_f1: %.4f | best_global_step: %d" % \
                                            (best_dev32_acc, best_dev32_f1, best_global_step))
                                logger.info(dev32_scores)

                                self.save(pattern_iter_output_dir)
                                logger.info("eval_data performance:")
                                eval_scores = self.eval_dev(eval_data, eval_config, n_gpu)
                                logger.info(eval_scores)
                            else:
                                early_stop_epoch += 1
                                logger.info(dev32_scores)
                                logger.info(early_stop_epoch)


                        elif self.config.task_name in ["rte", "wic", "boolq", "wsc", "copa"]:
                            if dev32_scores["acc"] >= best_dev32_acc:
                                if dev32_scores["acc"] > best_dev32_acc:
                                    early_stop_epoch = 0
                                else:
                                    early_stop_epoch += 1

                                best_dev32_acc = dev32_scores["acc"]
                                best_global_step = global_step
                                best_loss = tr_loss

                                logger.info("Saving trained model at {}...".format(pattern_iter_output_dir))
                                logger.info("best_dev32_acc: %.4f | best_global_step: %d" % \
                                            (best_dev32_acc, best_global_step))

                                self.save(pattern_iter_output_dir)
                                logger.info("eval_data performance:")
                                eval_scores = self.eval_dev(eval_data, eval_config, n_gpu)
                                logger.info(eval_scores)
                            else:
                                early_stop_epoch += 1
                                logger.info(dev32_scores)
                                logger.info(early_stop_epoch)

                if 0 < max_steps < global_step or early_stop_epoch >= 10:
                    epoch_iterator.close()
                    break

            if 0 < max_steps < global_step or early_stop_epoch >= 10:
                train_iterator.close()
                break

        return best_global_step, (best_loss / best_global_step if best_global_step > 0 else -1)




    def eval_dev(self, dev_data, eval_config, n_gpu, 
             encoding = None, 
             concept2cls = None, 
             num_concepts = -1,
             num_images_per_class = -1,
             submodular_weights = None,r=None,not_pt=None,vocab_embeds = None, vocab = None):
        self.model.eval()
        results = self.eval(dev_data,
                            per_gpu_eval_batch_size=eval_config.per_gpu_eval_batch_size,
                            n_gpu=n_gpu, encoding = encoding, 
                            concept2cls = concept2cls, 
                            num_concepts = num_concepts,
                            num_images_per_class = num_images_per_class,
                            submodular_weights = submodular_weights, r = r, not_pt=not_pt, vocab_embeds = vocab_embeds, vocab = vocab)
        predictions = np.argmax(results['logits'], axis=1)
        scores = {}
        metrics = eval_config.metrics if eval_config.metrics else ['acc']
        for metric in metrics:
            if metric == 'acc':
                scores[metric] = simple_accuracy(predictions, results['labels'])
            elif metric == 'f1':
                scores[metric] = f1_score(results['labels'], predictions)
            elif metric == 'f1-macro':
                scores[metric] = f1_score(results['labels'], predictions, average='macro')
            elif metric == 'em':
                scores[metric] = exact_match(predictions, results['labels'], results['question_ids'])
            else:
                raise ValueError(f"Metric '{metric}' not implemented")
        return scores



    @torch.no_grad()
    def interprete(self, eval_data: List[InputExample],
             per_gpu_eval_batch_size: int = 8,
             n_gpu: int = 1,
             encoding = None, 
             concept2cls = None, 
             num_concepts = -1,
             num_images_per_class = -1,
             submodular_weights = None,
             r = None,not_pt=None,vocab_embeds = None, vocab = None):
        eval_dataset = self._generate_dataset(eval_data)
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        preds = None
        all_indices, out_label_ids, question_ids = None, None, None
        interpret_book = {"global_p1":[],"global_p2":[],"idx":[], "p1":[],"p2":[],"label":[]}
        model = self.model.module if hasattr(self.model, 'module') else self.model
        score = model.score # NQ,NC
        _, score_indices = torch.topk(score, k = 2, dim = 1) # NQ,2
        score_indices = score_indices.cpu().numpy().tolist()
        # print(score_indices) 
        interpret_book["global_p1"].extend(score_indices[0])
        interpret_book["global_p2"].extend(score_indices[1])
        num_class = len(concept2cls) // 50
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = {k: t.cuda() for k, t in batch.items()}
            labels = batch['labels']
            indices = batch['idx']
            # if not_pt is None:
            
            assert len(labels) == 1
            category = labels[0].item()
            interpret_book["label"].append(category)
            step = num_concepts // num_class
            k = 50
            if self.config.submodular:
                text_feat = self.model.encode(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"]) # B,D
                concept_feat = self.model.encode(input_ids = encoding["input_ids"], attention_mask = encoding["attention_mask"]) 
                select_idx = None
                # if  r is None:
                select_idx =  submodular_select(text_feat, concept_feat,  concept2cls, num_concepts, num_images_per_class, submodular_weights)
                # print(select_idx)
                # select_concept_feat = concept_feat[select_idx] # N_c, D
                concept_feat = concept_feat[category * k : (category + 1) * k, :]
                sim = text_feat @ concept_feat.T # B,C
                # replace_embeds = model.prompt_embeddings(
                # torch.LongTensor(list(range(model.prompt_length))).cuda()) # [NQ,D]
                # batch["concept"] = select_concept_feat
                # filtering

                # sim = text_feat @ select_concept_feat.T # B,num_concepts
                # sim = select_concept_feat @ replace_embeds.T # N_c, N_q Internal pattern

                # context_sim = select_concept_feat @ text_feat.T # N_c,B
                
                # top2_indices = sim.topk()
                # _,prompt_indices = torch.topk(sim, k = 2, dim = 0) # 2,N_q
                # _,context_indices = torch.topk(context_sim, k = 2, dim = 0) # 2,B
                # print(context_sim.cpu().numpy().tolist())
                # print(context_indices)
                # print(category)
                # print(select_idx)
                # candidate = select_idx[category * step : (category + 1) * step ] # step  
                # print(candidate)
                # prompt_coef = score[:,candidate] # NQ, step [0,9]
                # prompt_coef = score[:, category * step : (category + 1) * step] # step
                # prompt_coef = prompt_coef[0,:]
                # idNc2step = {x : i for i, x in  enumerate(candidate)} 
                # idstep2Nc = {i : x for i, x in  enumerate(candidate)}
                # if labels.item() == 0:
                    # candidate = select_idx[:len(select_idx) // ]
                _,prompt_coef_top3 = torch.topk(sim, k = 3, dim = 1)
                per_batch_idx = indices.cpu().numpy().tolist()
                interpret_book["idx"].extend(per_batch_idx)
                # _, prompt_coef_top3 = torch.topk(prompt_coef, k = 3)
                prompt_coef_top3 = prompt_coef_top3[0].cpu().numpy().tolist()
                # top3 = [candidate[prompt_coef_top3[x]] for x in  range(len(prompt_coef_top3))]
                top3 = prompt_coef_top3
                # top3.append()
                # _, prompt_coef_top5 = torch.topk(prompt_coef, k = 5, dim = 1) # NQ,5

                # prompt_coef_top5 = prompt_coef_top5.cpu().numpy().tolist()
                # for y in range(len(prompt_coef_top2[x])) = [idstep2Nc[prompt_coef_top2[x][y]] for x in range(len(prompt_coef_top2)) for y in range(len(prompt_coef_top2[x]))]
                # print(prompt_coef_top2)

                # for x in range(len(prompt_coef_top5)):
                #      for y in range(len(prompt_coef_top5[x])):
                #         prompt_coef_top5[x][y] = candidate[prompt_coef_top5[x][y]]
                        
                # prompt1_top2_indices = prompt_coef_top2[0]
                # prompt2_top2_indices = prompt_coef_top2[1]

                # prompt_top1_indices = prompt_indices[0,:].cpu().numpy().tolist() # N_q
                # prompt_top2_indices = prompt_indices[1,:].cpu().numpy().tolist() 
                # context_top1_indices = context_indices[0,:].cpu().numpy().tolist() # B
                # context_top2_indices = context_indices[1,:].cpu().numpy().tolist()
                interpret_book["p1"].extend(top3)
                # for x in range(len(prompt_coef_top5)):
                #     for y in range(len(prompt_coef_top5[x])):
                #         if x == 0:
                #             interpret_book["p1"].append(prompt_coef_top5[x][y])
                #         else:
                #             interpret_book["p2"].append(prompt_coef_top5[x][y])
                # interpret_book["prompt1_concept_top1"].append(prompt_coef_top2[0][0])
                # interpret_book["prompt2_concept_top1"].append(prompt_coef_top2[1][0])
                # interpret_book["prompt1_concept_top2"].append(prompt_coef_top2[0][1])
                # interpret_book["prompt2_concept_top2"].append(prompt_coef_top2[1][1])

                # interpret_book["context_concept_top1"].extend(context_top1_indices)
                # interpret_book["context_concept_top2"].extend(context_top2_indices)
        
        return interpret_book

    def eval(self,
             eval_data: List[InputExample],
             per_gpu_eval_batch_size: int = 8,
             n_gpu: int = 1,
             encoding = None, 
             concept2cls = None, 
             num_concepts = -1,
             num_images_per_class = -1,
             submodular_weights = None,
             r = None,not_pt=None,vocab_embeds = None, vocab = None) -> Dict:

        eval_dataset = self._generate_dataset(eval_data)
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        preds = None
        all_indices, out_label_ids, question_ids = None, None, None
        eval_losses = [0.0]
        if vocab is not None:
            tok_vocab = [self.tokenizer.encode(x, add_special_tokens=False)[0] for x in vocab]
            cur_model = self.model.module if hasattr(self.model, 'module') else self.model
            embeds = cur_model.model.bert.embeddings.word_embeddings
            vocab_embeds = embeds.weight.data[tok_vocab] 
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = {k: t.cuda() for k, t in batch.items()}
            labels = batch['labels']
            indices = batch['idx']
            if not_pt is None:
                if self.config.submodular:
                    text_feat = self.model.encode(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"])
                    concept_feat = self.model.encode(input_ids = encoding["input_ids"], attention_mask = encoding["attention_mask"])
                    select_idx = None
                    if  r is None:
                        select_idx =  submodular_select(text_feat, concept_feat,  concept2cls, num_concepts, num_images_per_class, submodular_weights)
                        
                        select_concept_feat = concept_feat[select_idx] # num_concepts, D
                        batch["concept"] = select_concept_feat
                    else:
                        select_idx = sorted(random.sample(range(len(concept_feat)), num_concepts))
                        select_concept_feat = concept_feat[select_idx]
                        batch["concept"] = select_concept_feat
                    if self.args.train_bow != -1:
                        batch["bow_input"] = select_concept_feat
                if vocab_embeds is not None:
                    batch["vocab"] = vocab_embeds
            
            with torch.no_grad():

                logits = self.task_helper.eval_step(batch) if self.task_helper else None
                if not_pt is None:
                    if logits is None:
                        logits = EVALUATION_STEP_FUNCTIONS[MLM_WRAPPER](self)(batch)
                else:
                    _,logits = self.model(batch["input_ids"], batch["attention_mask"],batch["labels"])

                prediction_scores = logits.float().cuda()
                eval_loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))
                eval_losses.append(eval_loss.item())

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
                all_indices = indices.detach().cpu().numpy()
                if 'question_idx' in batch:
                    question_ids = batch['question_idx'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
                all_indices = np.append(all_indices, indices.detach().cpu().numpy(), axis=0)
                if 'question_idx' in batch:
                    question_ids = np.append(question_ids, batch['question_idx'].detach().cpu().numpy(), axis=0)


        return {
            "eval_loss": np.mean(eval_losses),
            'indices': all_indices,
            'logits': preds,
            'labels': out_label_ids,
            'question_ids': question_ids
        }

    def eval_AOPC(self,
                eval_data: List[InputExample],
                per_gpu_eval_batch_size: int = 8,
                n_gpu: int = 1,
                encoding = None, 
                concept2cls = None, 
                num_concepts = -1,
                num_images_per_class = -1,
                submodular_weights = None,
                eval_method = None, topk = 5, vocab = None, prefix = None):
        
        eval_dataset = self._generate_dataset(eval_data)
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        
        # load checkpoint
        
        ckpt_path = os.path.join(prefix, "weight.npy")
        ckpt = np.load(ckpt_path)
        # with open(prefix, )
        # deltas1, deltas3, deltas5 = [], [], []
        # deltas3, deltas5,delta10 = [], [], []
        deltas = [[], [], []] # top3, top5,top10
        topks = [3,5,10]
        # concept
        cur_score = self.model.score.cpu().numpy()
        for  _, topk in enumerate(topks):
            cur_score_topk = cur_score[:, :topk].flatten() 
            ckpt_topk = ckpt[:, :topk].flatten() 
            # cur_score_topk = cur_score
            # print(cur_score_topk, ckpt_topk)
            # print(np.corrcoef(cur_score_topk[:3], ckpt_topk))
            # print(cur_score_topk[:3])
            # print(ckpt_topk)
            a = np.corrcoef(cur_score_topk[:topk], ckpt_topk)[0,1]
            b = np.corrcoef(cur_score_topk[topk:],ckpt_topk)[0,1]
            # if self.args.pattern_ids == 2:
            corr_topk = b
            # else:
                # corr_topk = max(a, b)
            deltas[_].append(corr_topk)
            # random
            # random_cef = np.ones(topk) / topk
            random_cef = np.random.rand(topk)
            corr_random_topk = np.corrcoef(random_cef, ckpt_topk)[0,1]
            # print(f"random_cef:{random_cef} corr:{corr_random_topk} ")
            deltas[_].append(corr_random_topk)
            # continue
        return { "top3" : deltas[0], "top5" : deltas[1], "top10" : deltas[2]}
        # exit(1s)
        # return None

        deltas_ig, deltas_grad =  [[],[],[]], [[],[],[]]
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = {k: t.cuda() for k, t in batch.items()}
            labels = batch['labels']
            indices = batch['idx']
            if self.config.submodular:
                text_feat = self.model.encode(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"])
                concept_feat = self.model.encode(input_ids = encoding["input_ids"], attention_mask = encoding["attention_mask"])
                select_idx = None
                select_idx =  submodular_select(text_feat, concept_feat,  concept2cls, num_concepts, num_images_per_class, submodular_weights)

                select_concept_feat = concept_feat[select_idx] # num_concepts, D
                batch["concept"] = select_concept_feat
                
            # with torch.no_grad():
                
            #     logits = self.task_helper.eval_step(batch) if self.task_helper else None
            #     if logits is None:
            #         logits = EVALUATION_STEP_FUNCTIONS[MLM_WRAPPER](self)(batch)
            #         logits = softmax(logits, -1)

                
            indices = None
            if eval_method == "concept":
                if self.model.score.shape[0] == 2:
                    _, indices = torch.sum(torch.abs(self.model.score), dim = 0).topk(self.config.topk, dim = 0)
                    indices = indices.detach().cpu().numpy().tolist()
                elif self.model.score.shape[0] == 1:
                    _, indices = self.model.score.topk(self.config.topk, dim = 1)
                    indices = indices[0,:].detach().cpu().numpy().tolist()
                
            # elif eval_method == "random":
                # indices = random.sample(list(range(self.model.score.shape[1])), self.config.topk)
                # cur_sample_cef = np.ones()
            # elif eval_method == "ig" or eval_method == "grad":
            elif eval_method == "corr":
                # if eval_method == "ig":
                    concept_ig = integrated_gradients(self, batch, encoding)
                    concept_select_ig = concept_ig[select_idx]
                    cur_sample_cef_ig = concept_select_ig.detach().cpu().numpy().flatten()
                    
                    # _,indices = concept_select_ig.topk(k = topk, dim = 0)
                # else:
                    grad = gradients(self, batch, encoding)
                    concept_select_grad = grad[select_idx]
                    cur_sample_cef_grad = concept_select_grad.detach().cpu().numpy().flatten()
                    # _,indices = concept_select_grad.topk(k = topk, dim = 0)


            elif eval_method == "word":
                tok_vocab = [self.tokenizer.encode(x, add_special_tokens=False)[0] for x in vocab]
                embeds = self.model.model.bert.embeddings.word_embeddings
                vocab_embeds = embeds.weight.data[tok_vocab] # 1500,1024
                if self.model.r.shape[0] == 2:
                    _, indices = torch.sum(self.model.r, dim = 0).topk(self.config.topk, dim = 0)
                    indices = indices.detach().cpu().numpy().tolist()
                elif self.model.r.shape[0] == 1:
                    _, indices = self.model.r.topk(self.config.topk, dim = 1)
                    indices = indices[0,:].detach().cpu().numpy().tolist()
                
            for _, topk in enumerate(topks):
                # print(cur_sample_cef_ig.shape, cur_score[:,:topk].shape) (10,) (2,3)
                a = np.corrcoef(cur_sample_cef_ig[:topk], cur_score[0,:topk].flatten())[0,1]
                b = np.corrcoef(cur_sample_cef_ig[:topk], cur_score[1,:topk].flatten())[0,1]
                deltas_ig[_].append(max(a,b))
                
                # deltas_ig[_].append(np.corrcoef(cur_sample_cef_ig, cur_score[:,:topk].flatten())[0,1])
                a = np.corrcoef(cur_sample_cef_grad[:topk], cur_score[0,:topk].flatten())[0,1]
                b = np.corrcoef(cur_sample_cef_grad[:topk], cur_score[1,:topk].flatten())[0,1]
                deltas_grad[_].append(max(a,b))
                # deltas_grad[_].append(np.corrcoef(cur_sample_cef_grad, cur_score[:,:topk].flatten())[0,1])
            # with torch.no_grad():
            #     for topk in [1,3,5]:
            #         for aopc_step in range(topk):
            #             step_indices = indices[:aopc_step+1]
            #             batch["position"] = step_indices
            #             if eval_method == "word":
            #                 batch["vocab"] = vocab_embeds
            #             logits_step = self.mlm_eval_step(batch)
            #             logits_step = softmax(logits_step, -1)
                        
            #             delta_p = logits[torch.arange(logits.shape[0]),batch["labels"]] - logits_step[torch.arange(logits_step.shape[0]),batch["labels"]] # B
                        
            #             if topk == 1:
            #                 deltas1.append(delta_p.mean().item()) #1,B
            #             elif topk == 3:
            #                 deltas3.append(delta_p.mean().item())
            #             else:
            #                 deltas5.append(delta_p.mean().item())
                    
        for _, topk in enumerate(topks):
            d1 = np.mean(deltas_ig[_])
            deltas[_].append(d1)
            d2 = np.mean(deltas_grad[_])
            deltas[_].append(d2)
             

        # deltas1 = np.mean(deltas1)
        # deltas3 = np.mean(deltas3)
        # deltas5 = np.mean(deltas5)
        # return {
        #     "AOPC1" : deltas1, "AOPC3" : deltas3, "AOPC5" : deltas5
        # }
        return { "top3" : deltas[0], "top5" : deltas[1], "top10" : deltas[2]}

                


    def _generate_dataset(self, data: List[InputExample], labelled: bool = True):
        features = self._convert_examples_to_features(data, labelled=labelled)
        
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            'labels': torch.tensor([f.label for f in features], dtype=torch.long),
            'mlm_labels': torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
            'logits': torch.tensor([f.logits for f in features], dtype=torch.float),
            'idx': torch.tensor([f.idx for f in features], dtype=torch.long),
            'block_flag': torch.tensor([f.block_flag for f in features], dtype=torch.long)
        }

        if self.task_helper:
            self.task_helper.add_features_to_dict(features, feature_dict)
        return DictDataset(**feature_dict)


    def _convert_examples_to_features(self, examples: List[InputExample], labelled: bool = True) -> List[InputFeatures]:
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example {}".format(ex_index))
            input_features = self.preprocessor.get_input_features(example, labelled=labelled)
            if self.task_helper:
                self.task_helper.add_special_input_features(example, input_features)
            features.append(input_features)
            """
            if ex_index < 5:
                logger.info(f'--- Example {ex_index} ---')
                logger.info(input_features.pretty_print(self.tokenizer))
            """
        return features


    def generate_default_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        input_ids = batch['input_ids']
        bz = batch['input_ids'].shape[0]
        block_flag = batch["block_flag"]
        
        model = self.model.module if hasattr(self.model, 'module') else self.model

        if self.config.model_type == "albert":
            raw_embeds = model.model.albert.embeddings.word_embeddings(input_ids)
        elif self.config.model_type == "bert":
            raw_embeds = model.model.bert.embeddings.word_embeddings(input_ids)
        elif self.config.model_type == "roberta":
            raw_embeds = model.model.roberta.embeddings.word_embeddings(input_ids)
        elif self.config.model_type == "GPT2":
            raw_embeds = model.model.transformer.wte(input_ids)
        
        if "concept" not in batch.keys() and "vocab" not in batch.keys(): # ptuning
            replace_embeds = model.prompt_embeddings(
                torch.LongTensor(list(range(model.prompt_length))).cuda()) # [NQ,D]

            replace_embeds = replace_embeds.unsqueeze(0) # [batch_size, NQ, embed_size]
        elif "concept" in batch.keys() and "bow_input" not in batch.keys(): # concept
            select_concept = batch["concept"] # NC,emb_size
            if "position" in batch.keys():
                select_concept[batch["position"], :] = 0 # Nq,nc @ NC,D
            
            replace_embeds = self.model.score @ select_concept  # NQ, D
            replace_embeds = replace_embeds.unsqueeze(0) # 1, NQ, embed_size
        elif "concept" in batch.keys() and "bow_input"  in batch.keys(): # bow
            select_concept = batch["concept"] # NC,emb_size
            replace_embeds = self.model.bow(select_concept) # NQ, D
            replace_embeds = replace_embeds.unsqueeze(0) # 1, NQ, embed_size

        elif "vocab" in batch.keys(): # word
            vocab_embs = batch["vocab"] # 1500,1024
            if "position" in batch.keys():
                vocab_embs[batch["position"] , :] = 0

            replace_embeds = self.model.r @ vocab_embs # NQ,1024
            replace_embeds = replace_embeds.unsqueeze(0)
        

            
            
        if self.config.prompt_encoder_type == "lstm":
            replace_embeds = model.lstm_head(replace_embeds)[0]  # [batch_size, seq_len, 2 * hidden_dim]
            if model.prompt_length == 1:
                replace_embeds = model.mlp_head(replace_embeds)
            else:
                replace_embeds = model.mlp_head(replace_embeds).squeeze()

        elif self.config.prompt_encoder_type == "mlp":
            replace_embeds = model.mlp(replace_embeds)
        else:
            raise ValueError("unknown prompt_encoder_type.")
        
        
        
        blocked_indices = (block_flag == 1).nonzero().reshape((bz, model.prompt_length, 2))[:, :, 1]
        
        for bidx in range(bz):
            for i in range(blocked_indices.shape[1]):
                # print(blocked_indices[bidx, i])
                # print(raw_embeds.shape, replace_embeds.shape)
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]

        inputs = {'inputs_embeds': raw_embeds, 'attention_mask': batch['attention_mask']}

        if self.config.model_type in ['bert']:
            inputs['token_type_ids'] = batch['token_type_ids']

        return inputs


    def mlm_train_step(self, labeled_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM training step."""
        inputs = self.generate_default_inputs(labeled_batch)
        mlm_labels, labels = labeled_batch['mlm_labels'], labeled_batch['labels']
        outputs = self.model(**inputs)
        prediction_scores = self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(mlm_labels, outputs[0])
        loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))

        return loss


    def mlm_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM evaluation step."""
        inputs = self.generate_default_inputs(batch)
        outputs = self.model(**inputs)
        return self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs[0])

    def mlm_eval_step2(self,batch: Dict[str, torch.Tensor], inputs: torch.Tensor):
        outputs = self.model(inputs_embeds = inputs)
        return self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs[0])
