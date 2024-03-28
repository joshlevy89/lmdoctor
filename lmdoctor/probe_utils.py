import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
from collections import defaultdict


def scale_direction(act_pairs, direction):
    # scale direction such that p(mu_b + scaled_direction) = p(mu_a), following marks et al
    # (https://arxiv.org/abs/2310.06824)
    mu_a = torch.mean(act_pairs[:, 0, :], axis=0)
    mu_b = torch.mean(act_pairs[:, 1, :], axis=0)
    norm_direction = F.normalize(direction, dim=0)
    diff_proj = (mu_a - mu_b) @ norm_direction.view(-1)
    scaled_direction = (norm_direction * diff_proj).view(-1)
    return scaled_direction
    

def probe_pca(train_acts, device):
    directions = {}
    scaled_directions = {}
    mean_diffs = {}
    direction_info = defaultdict(dict)
    for layer in train_acts:
        act_pairs = train_acts[layer]
        shuffled_pairs = [] # must shuffle pairs to create variability in difference between pairs
        for pair in act_pairs:
            pair = pair[torch.randperm(2)]
            shuffled_pairs.append(pair)
        shuffled_pairs = torch.stack(shuffled_pairs)
        diffs = shuffled_pairs[:, 0, :] - shuffled_pairs[:, 1, :] 
        mean_diffs[layer] = torch.mean(diffs, axis=0)
        centered_diffs = diffs - mean_diffs[layer] # is centering necessary?
        pca = PCA(n_components=1)
        pca.fit(centered_diffs.detach().cpu())
        direction = torch.tensor(pca.components_[0], dtype=act_pairs.dtype).to(device)
        directions[layer] = direction 
        # scale
        act_pairs = train_acts[layer]
        scaled_direction = scale_direction(act_pairs, direction)
        scaled_directions[layer] = scaled_direction
    
    direction_info['unscaled_directions'] = directions # these aren't used, but kept for posterity
    direction_info['directions'] = scaled_directions
    direction_info['mean_diffs'] = mean_diffs
    return direction_info


def probe_logreg(train_acts, device):

    def _train_logreg(X, y, test_clf=False):    
        if test_clf:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            clf = LogisticRegression(random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)    
            print(classification_report(y_test, y_pred))
            
        # fit on all data
        clf = LogisticRegression(random_state=42)
        clf.fit(X, y)
    
        return clf

    directions = {}
    scaled_directions = {}
    direction_info = defaultdict(dict)
    for layer in train_acts:
        act_pairs = train_acts[layer]
        n_pairs, n_classes, hidden_dim = act_pairs.shape
        acts = act_pairs.view(-1, hidden_dim)
        labels = [1, 0] * n_pairs
        clf = _train_logreg(acts.cpu().numpy(), np.array(labels))
        direction = torch.tensor(clf.coef_[0], dtype=act_pairs.dtype).to(device)
        directions[layer] = direction
        # scale
        scaled_direction = scale_direction(act_pairs, direction)
        scaled_directions[layer] = scaled_direction

    direction_info['unscaled_directions'] = directions # these aren't used, but kept for posterity
    direction_info['directions'] = scaled_directions
    return direction_info


def probe_massmean(train_acts, device):
    directions = {}
    scaled_directions = {}
    direction_info = defaultdict(dict)
    for layer in train_acts:
        act_pairs = train_acts[layer]
        mu_a = torch.mean(act_pairs[:, 0, :], axis=0)
        mu_b = torch.mean(act_pairs[:, 1, :], axis=0)
        direction = mu_a - mu_b
        directions[layer] = direction
        # scale (technically, don't need to scale bc it's already scaled, but doing for consistency)
        scaled_direction = scale_direction(act_pairs, direction)
        scaled_directions[layer] = scaled_direction  

    direction_info['unscaled_directions'] = directions # these aren't used, but kept for posterity
    direction_info['directions'] = scaled_directions
    return direction_info