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

import ast
import json
import os
import statistics
from collections import defaultdict
from typing import List, Dict

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers.data.metrics import simple_accuracy

import logging as log
from pet.config import EvalConfig, TrainConfig
from pet.utils import InputExample, exact_match, save_logits, save_predictions, softmax, LogitsList, set_seed, eq_div
from pet.wrapper import TransformerModelWrapper
from pet.config import  WrapperConfig
import torch
logger = log.getLogger('root')




def init_model(config: WrapperConfig, not_pt = None, vocab = None, all_args = None) -> TransformerModelWrapper:
    """Initialize a new model from the given config."""
    assert config.pattern_id is not None, 'A pattern_id must be set for initializing a new PET model'
    model = TransformerModelWrapper(config, not_pt, vocab1500 = vocab, all_args = all_args)
    return model


def train_pet(train_data: List[InputExample],
              eval_data: List[InputExample],
              dev32_data: List[InputExample],
              model_config: WrapperConfig,
              train_config: TrainConfig,
              eval_config: EvalConfig,
              pattern_ids: List[int],
              output_dir: str,
              train_mode = None,
              shot_number = None,
              repetitions: int = 3,
              do_train: bool = True,
              do_eval: bool = True,
              seed: int = 42,
              encoding = None,
              submodular = False,
              label_set = None,
              concept2cls = None,
              num_concepts = -1,
              num_images_per_class = -1,
              submodular_weights = None,
              r = None, eval_method = None, topk = None, not_pt = None, vocab1500 = None, all_args = None
              ):

    """
    Train and evaluate a new PET model for a given task.

    :param model_config: the model configuration for each model corresponding to an individual PVP
    :param train_config: the training configuration for each model corresponding to an individual PVP
    :param eval_config: the evaluation configuration for each model corresponding to an individual PVP
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param repetitions: the number of training repetitions for each model corresponding to an individual PVP
    :param train_data: the training examples to use
    :param dev32_data: the dev32 examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """

    results = defaultdict(lambda: defaultdict(list))
    dev32_results = defaultdict(lambda: defaultdict(list))
    set_seed(seed)


    model_config.pattern_id = pattern_ids
    model_config.submodular = submodular
    results_dict = {}

    if not_pt is None:      
        pattern_iter_output_dir = "{}/{}/{}shot/{}".format(output_dir, train_mode, shot_number,seed)
        if eval_method is not None:
            pattern_iter_output_dir = "{}/{}/{}shot/top{}".format(output_dir, train_mode, shot_number, model_config.topk)
    else:
        # medical fine tuning
        pattern_iter_output_dir = "{}/ft/{}shot/{}".format(output_dir, shot_number, seed)
    if all_args.train_bow != -1:
        pattern_iter_output_dir = "{}/BOW/{}shot/{}".format(output_dir, shot_number, seed)
    if vocab1500 is not None:
        pattern_iter_output_dir = "{}/baseline/{}shot/{}".format(output_dir, shot_number, seed)
    if eval_method == "corr":
        pattern_iter_output_dir = "{}/corr/{}shot".format(output_dir, shot_number)
    elif eval_method == "vis":
        pattern_iter_output_dir = f"{output_dir}/vis"
    prefix = "{}/BOW/{}shot/{}/ckpt".format(output_dir, shot_number,seed)
    if all_args.task_name == "agnews":
        prefix = "{}/BOW/{}shot/{}/ckpt".format(output_dir, -1,seed)
    if not os.path.exists(pattern_iter_output_dir):
        os.makedirs(pattern_iter_output_dir)
    
    wrapper = init_model(model_config, not_pt, vocab = vocab1500, all_args = all_args)
    print("ft shot_number:{} seed:{} step:{} test_result:{}".format(shot_number, seed, all_args.pet_max_steps, str(results_dict) ))

    # Training
    if do_train:


        results_dict.update(train_single_model(train_data, eval_data, dev32_data, pattern_iter_output_dir, \
                                                wrapper, train_config, eval_config, \
                                                encoding = encoding, \
                                                submodular = submodular, label_set = label_set, concept2cls=concept2cls, \
                                                num_concepts = num_concepts, num_images_per_class = num_images_per_class, \
                                                submodular_weights = submodular_weights,r = r, not_pt=not_pt, \
                                                vocab1500 = vocab1500, all_args = all_args)) 
        # if (train_mode == "concept" and pattern_ids == 2 and (shot_number == -1 or shot_number == 22500)) or all_args.train_bow != -1 or \
        # (train_mode == "ptuning" and pattern_ids == 2 and shot_number == -1):
        #     save_path = os.path.join(pattern_iter_output_dir, "ckpt")
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)

        #     wrapper.save(save_path)
        #     print("Save ckpt at {}".format(save_path))



        torch.cuda.empty_cache()

            # Evaluation
    if do_eval:
        print("Starting evaluation...")
        
        if not do_train:
            # shot_number = -1
            save_path = os.path.join("{}/{}/{}shot/{}".format(output_dir, train_mode, shot_number,seed), "ckpt")

            wrapper = TransformerModelWrapper.from_pretrained(save_path)
        
        if eval_method != "vis":           
            eval_result = evaluate(wrapper, eval_data, eval_config,
                                    encoding = encoding,           
                                    concept2cls = concept2cls,     
                                    num_concepts = num_concepts,   
                                    num_images_per_class = num_images_per_class, 
                                    submodular_weights = submodular_weights, r = r, \
                                    eval_method=eval_method, topk=topk, not_pt = not_pt, vocab1500 = vocab1500, prefix = prefix)
 
        if eval_method is None:
            # acc
            results_dict['eval_set_after_training'] = eval_result['scores']
            if all_args.train_bow == -1:
                if not_pt is None:
                    print("train_mode:{} shot_number:{} seed:{} pattern_id:{} step:{} test result:{}".format(train_mode, shot_number, seed, pattern_ids, all_args.pet_max_steps,str(results_dict)))
                    if vocab1500 is not None and len(vocab1500) != 1500:
                        write_path = os.path.join(pattern_iter_output_dir, 'results-wordnum{}-seed{}-pattern{}.txt'.format(len(vocab1500),seed,pattern_ids))
                        with open(write_path, "w") as fh:
                            fh.write(str(results_dict))
                    with open(os.path.join(pattern_iter_output_dir, 'results-seed{}-pattern{}-concept{}-step{}.txt'.format(seed,pattern_ids, all_args.num_concepts,all_args.pet_max_steps)), 'w') as fh:
                        fh.write(str(results_dict))
                else:
                    print("ft shot_number:{} seed:{} step:{} test_result:{}".format(shot_number, seed, all_args.pet_max_steps, str(results_dict) ))
                    with open(os.path.join(pattern_iter_output_dir, 'results-seed{}-step{}.txt'.format(seed, all_args.pet_max_steps)), 'w') as fh:
                        fh.write(str(results_dict))
            else:
        
                print('BOW shot_number:{} seed:{} pattern_id:{} test result:{}'.format(shot_number, seed, pattern_ids, str(results_dict)))
                with open(os.path.join(pattern_iter_output_dir, 'results-seed{}-pattern{}.txt'.format(seed,pattern_ids)), 'w') as fh:
                    fh.write(str(results_dict))
            
        elif eval_method == "corr":
            # corr/aopc
            # print("seed:{} test result:{}".format(seed, str(eval_result)))
            # with open(os.path.join(pattern_iter_output_dir, 'results-{}-seed{}-pattern{}-softmax.txt'.format(eval_method, seed,pattern_ids)), 'w') as fh:
            #     fh.write(str(eval_result))
            
            print("seed:{} test result:".format(seed))
            for topk in eval_result.keys():
                if len(eval_result[topk]) == 4:
                    print("top:{} concept:{:.3f} random:{:.3f} ig:{:.3f} grad:{:.3f}".format(topk, eval_result[topk][0], eval_result[topk][1], eval_result[topk][2], eval_result[topk][3]))
                else:
                    print("top:{} concept:{:.3f}".format(topk, eval_result[topk][0]))
            # bert-large/sst2/corr/-1shot/
            # if all_args.pattern_ids == 2:
            with open(os.path.join(pattern_iter_output_dir, 'results-pattern{}-{}.txt'.format(pattern_ids, seed)), 'w') as fh:
                fh.write(str(eval_result))
            # else:
            #     with open(os.path.join(pattern_iter_output_dir, 'results-pattern{}-{}.txt'.format(pattern_ids, seed)), 'w') as fh:
            #         fh.write(str(eval_result))
        elif eval_method == "vis":
            # interpret_book = {"idx":[], "prompt1_concept_top1":[], "prompt1_concept_top2":[], "prompt2_concept_top1":[],"prompt2_concept_top2":[], "context_concept_top1":[],"context_concept_top2":[]}

            inter_result =  wrapper.interprete(eval_data,
                        per_gpu_eval_batch_size = eval_config.per_gpu_eval_batch_size,
                        
                        encoding = encoding, 
                        concept2cls = concept2cls, 
                        num_concepts = num_concepts,
                        num_images_per_class = num_images_per_class,
                        submodular_weights = submodular_weights)
            label_set = encoding["text"]
            with open(os.path.join(pattern_iter_output_dir, f"interpret-seed{seed}.txt"),"w") as fw:
                fw.write("global\n")
                p1_top1 = label_set[inter_result["global_p1"][0]]
                p1_top2 = label_set[inter_result["global_p1"][1]]
                fw.write(f"p1_top1:{p1_top1} \n p1_top2:{p1_top2} \n")
                p2_top1 = label_set[inter_result["global_p2"][0]]
                p2_top2 = label_set[inter_result["global_p2"][1]]
                fw.write(f"p2_top1:{p2_top1} \n p2_top2:{p2_top2}\n")
                # for nid, p1_top1,p1_top2,p2_top1,p2_top2 in zip(inter_result["idx"],inter_result["prompt1_concept_top1"],inter_result["prompt1_concept_top2"],inter_result["prompt2_concept_top1"],inter_result["prompt2_concept_top2"]):
                j = 0
                step = 3
                k = 50
                for i, nid in enumerate(inter_result["idx"]):
                    # if nid >= len(eval_data):
                    #     break
                    fw.write(f"ID:{nid}\n")
                    fw.write(f"sample:{eval_data[nid].text_a}\n")
                    fw.write(f"Label:{eval_data[nid].label}\n")
                    stride =  inter_result["label"][i] * k
                    # print(stride)
                    # print(inter_result["p1"])
                    # print(inter_result["p1"][step * i])
                    fw.write("p1_top1:{}\n".format(label_set[inter_result["p1"][step * i] + stride ]))
                    fw.write("p1_top2:{}\n".format(label_set[inter_result["p1"][ 1 + step * i ] + stride]))
                    fw.write("p1_top3:{}\n".format(label_set[inter_result["p1"][ 2 + step * i] + stride]))
                    # fw.write("p1_top4:{}\n".format(label_set[inter_result["p1"][j + 3 + step * i]]))
                    # fw.write("p1_top5:{}\n".format(label_set[inter_result["p1"][j + 4 + step * i]]))

                    # fw.write("p2_top1:{}\n".format(label_set[inter_result["p2"][j + step * i]]))
                    # fw.write("p2_top2:{}\n".format(label_set[inter_result["p2"][j + 1 + step * i]]))
                    # fw.write("p2_top3:{}\n".format(label_set[inter_result["p2"][j + 2 + step * i]]))
                    # fw.write("p2_top4:{}\n".format(label_set[inter_result["p2"][j + 3 + step * i]]))
                    # fw.write("p2_top5:{}\n".format(label_set[inter_result["p2"][j + 4 + step * i]]))
                    # j += step
                    
                    fw.write("*" * 30)
                    fw.write("\n")
def train_single_model(train_data: List[InputExample],
                       eval_data: List[InputExample],
                       dev32_data: List[InputExample],
                       pattern_iter_output_dir: str,
                       model: TransformerModelWrapper,
                       config: TrainConfig,
                       eval_config: EvalConfig,
                       encoding = None,
                       submodular:bool = False,
                       label_set:List[str] = None,
                       concept2cls:List[int] = None,
                       num_concepts = -1,
                       num_images_per_class = -1,
                       submodular_weights = None,
                       r = None, not_pt = None, vocab1500 = None, all_args = None):
    """
    Train a single model.
    :param model: the model to train
    :param train_data: the training examples to use
    :param config: the training config
    :param eval_config: the evaluation config
    :return: a dictionary containing the global step, average loss and (optionally) results on the train set
    """

    results_dict = {}

    if not train_data:
        logger.warning('Training method was called without training examples')
    else:
        global_step, tr_loss = model.train(
            pattern_iter_output_dir=pattern_iter_output_dir,
            eval_config=eval_config,
            train_data=train_data,
            dev32_data=dev32_data,
            eval_data=eval_data,
            per_gpu_train_batch_size=config.per_gpu_train_batch_size,
            n_gpu=config.n_gpu,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            alpha=config.alpha,
            label_set=label_set,
            concept2cls = concept2cls,
            num_concepts = num_concepts,
            num_images_per_class = num_images_per_class,
            submodular_weights = submodular_weights,
            r = r, not_pt = not_pt, vocab1500 = vocab1500, word_lr = config.word_lr
        )
        results_dict['global_step'] = global_step
        results_dict['average_loss'] = tr_loss
   
    with torch.no_grad():
        results_dict['train_set_after_training'] = evaluate(model, train_data, eval_config, encoding = encoding, concept2cls = concept2cls, num_concepts = num_concepts, num_images_per_class = num_images_per_class, submodular_weights = submodular_weights, r = r, not_pt = not_pt, vocab1500 = vocab1500)['scores']['acc']
    return results_dict


def evaluate(model: TransformerModelWrapper,
             eval_data: List[InputExample],
             config: EvalConfig,
             encoding = None, 
             concept2cls = None, 
             num_concepts = -1,
             num_images_per_class = -1,
             submodular_weights = None, r = None, eval_method = None, 
             topk = 5, not_pt = None, vocab1500 = None, prefix = None) -> Dict:

    # metrics = config.metrics if config.metrics else ['acc']
    metrics = ["acc"]
    if eval_method == "corr":
        metrics = ["corr","aopc"]
    elif eval_method == "vis":
        metrics = ["vis"]
    # metrics = ["aopc"] if eval_method is not None else ["acc"]
    if "corr" in metrics:
        results = model.eval_AOPC(eval_data=eval_data,
                                  per_gpu_eval_batch_size=config.per_gpu_eval_batch_size,
                                  n_gpu=config.n_gpu,encoding=encoding,
                                  concept2cls = concept2cls, 
                                  num_concepts = num_concepts,
                                  num_images_per_class = num_images_per_class,
                                  submodular_weights = submodular_weights,
                                  eval_method=eval_method, topk = topk, vocab = vocab1500, prefix = prefix)
    elif "acc" in metrics :
        results = model.eval(eval_data=eval_data,
                            per_gpu_eval_batch_size=config.per_gpu_eval_batch_size,
                            n_gpu=config.n_gpu, encoding = encoding, 
                            concept2cls = concept2cls, 
                            num_concepts = num_concepts,
                            num_images_per_class = num_images_per_class,
                            submodular_weights = submodular_weights, r = r, not_pt= not_pt,vocab = vocab1500)
        predictions = np.argmax(results['logits'], axis=1)
        scores = {}
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
        results['scores'] = scores
        results['predictions'] = predictions
    elif "vis" in metrics:

        pass

    
        
    return results


def _write_results(path: str, all_results: Dict, dev32_results: Dict):
    with open(path, 'w') as fh:

        results = all_results
        logger.info("eval_results:")
        fh.write("eval_results:" + '\n')

        for metric in results.keys():
            for pattern_id, values in results[metric].items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                result_str = "{}-p{}: {} +- {}".format(metric, pattern_id, mean, stdev)
                logger.info(result_str)
                fh.write(result_str + '\n')

        for metric in results.keys():
            all_results = [result for pattern_results in results[metric].values() for result in pattern_results]
            all_mean = statistics.mean(all_results)
            all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
            result_str = "{}-all-p: {} +- {}".format(metric, all_mean, all_stdev)
            logger.info(result_str)
            fh.write(result_str + '\n')

        logger.info("dev32_results:")
        fh.write("dev32_results:" + '\n')

        for metric in dev32_results.keys():
            for pattern_id, values in dev32_results[metric].items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                result_str = "{}-p{}: {} +- {}".format(metric, pattern_id, mean, stdev)
                logger.info(result_str)
                fh.write(result_str + '\n')

        for metric in dev32_results.keys():
            all_results = [result for pattern_results in dev32_results[metric].values() for result in pattern_results]
            all_mean = statistics.mean(all_results)
            all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
            result_str = "{}-all-p: {} +- {}".format(metric, all_mean, all_stdev)
            logger.info(result_str)
            fh.write(result_str + '\n')

