from apricot import CustomSelection, MixtureSelection, FacilityLocationSelection
import torch as th

import numpy as np
from tqdm import tqdm
def bert_score(text_feat, concept_feat, num_images_per_class):
    # text_feat : B,D concept_feat: NC,D
    # num_images_per_class: C,K
    num_cls = len(num_images_per_class)
    scores_mean = th.empty((concept_feat.shape[0], num_cls)) # NC,C
    start_loc = 0
    for i in range(num_cls):
        end_loc = sum(num_images_per_class[:i+1])
        scores_mean[:, i] = (concept_feat @ text_feat[start_loc:end_loc].t()).mean(dim=-1)  # NC
        start_loc = end_loc
    return scores_mean # NC,C

def mi_score(text_feat, concept_feat,  num_images_per_class):
    # print(f"text_feat.shape:{text_feat.shape} concept_feat:{concept_feat.shape}")
    # text_feat.shape:torch.Size([32, 1024]) concept_feat:torch.Size([242, 1024])

    num_cls = len(num_images_per_class)
    scores_mean = bert_score(text_feat, concept_feat, num_images_per_class) # Sim(c,y)  (NC,C)
    # print(f"scores_mean.shape:{scores_mean.shape}")
    # scores_mean.shape:torch.Size([242, 5])

    normalized_scores = scores_mean / (scores_mean.sum(dim=0) * num_cls) # Sim_bar(c,y) (NC,C)
    # print(f"normalized_scores.shape:{normalized_scores.shape}")
    # normalized_scores.shape:torch.Size([242, 5])

    margin_x = normalized_scores.sum(dim=1) # sum_y in Y Sim_bar(c,y) # NC
    margin_x = margin_x.reshape(-1, 1).repeat(1, num_cls) # NC,C
    # compute MI and PMI
    pmi = th.log(normalized_scores / (margin_x * 1 / num_cls)) # log Sim_bar(c,y) / sum_y in Y Sim_bar(c,y) / N = log(Sim_bar(c|y)) # NC,C
    # print(f"pmi.shape:{pmi.shape}")
    # pmi.shape:torch.Size([242, 5])

    mi = normalized_scores * pmi  # Sim_bar(c,y)* log(Sim_bar(c|y))
    mi = mi.sum(dim=1) # NC
    return mi, scores_mean

def submodular_select(text_feat, concept_feat,  concept2cls, num_concepts, num_images_per_class, submodular_weights, *args):
    # text_feat: B,D
    # concept_feat : C*NC,D
    # concept2cls: NC
    # num_concepts: NC
    # num_images_per_class: [K,K]
    # submodular_weights:[Float,Float]
    assert num_concepts > 0
    # print(num_concepts, num_images_per_class)
    # num_cls = num_concepts // num_images_per_class # 20 // 5 = 4
    num_cls = len(concept2cls) // 50
    num_images_per_class = [num_images_per_class] * num_cls
    # num_images_per_class = [num_images_per_class] * 2
    # num_cls = len(num_images_per_class) 
    
    # num_cls = 2 
    
    all_mi_scores, _ = mi_score(text_feat, concept_feat,  num_images_per_class) # NC
    selected_idx = []
    num_concepts_per_cls = int(np.ceil(num_concepts / num_cls)) # 20 // 4 = 5

    def mi_based_function(X):
        return X[:, 0].sum()
    
    mi_selector = CustomSelection(num_concepts_per_cls, mi_based_function)
    distance_selector = FacilityLocationSelection(num_concepts_per_cls, metric='cosine')

    mi_score_scale = submodular_weights[0]
    facility_weight = submodular_weights[-1]
    
    if mi_score_scale == 0:
        submodular_weights = [0, facility_weight]
    else:
        submodular_weights = [1, facility_weight] 
    # submodular_weights [1, 5]
#     concept2cls = th.from_numpy(concept2cls).long()
    concept2cls = th.tensor(concept2cls).long()
#     concept2cls = th.LongTensor(concept2cls)
    
    for i in range(num_cls):
#         print(concept2cls)
        
        cls_idx = th.where(concept2cls == i)[0] 

        if len(cls_idx) <= num_concepts_per_cls:
            selected_idx.extend(cls_idx)
        else:
            # print(all_mi_scores.device, cls_idx.device)
            if cls_idx.device is not "cpu":
                cls_idx = cls_idx.detach().cpu()
            # print(all_mi_scores.shape)
            # print(cls_idx)
            # print(num_images_per_class)
            mi_scores = all_mi_scores[cls_idx] * mi_score_scale

            current_concept_features = concept_feat[cls_idx].detach().cpu()
#             print(mi_scores.device, current_concept_features.device) cpu cuda:0
            augmented_concept_features = th.hstack([th.unsqueeze(mi_scores, 1), current_concept_features]).numpy()
            selector = MixtureSelection(num_concepts_per_cls, functions=[mi_selector, distance_selector], weights=submodular_weights, optimizer='naive', verbose=False)
            
            selected = selector.fit(augmented_concept_features).ranking
            selected_idx.extend(cls_idx[selected])
    selected_idx = [x.item() for x in selected_idx]
    return sorted(selected_idx)