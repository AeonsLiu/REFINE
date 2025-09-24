import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from collections import Counter
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Filter:
    def __init__(self, X, y, sparse_labels: bool = False, catboost: bool = False):
        self.X = X
        self.y = np.asarray(y)
        self._sparse_labels = sparse_labels
        self._gold_labels_probabilities = None
        self._true_probabilities = None
        self.catboost = catboost

    def on_epoch_end(self, clf, device="cpu", iteration=1, **kwargs):
        x = self.X
        y = torch.tensor(self.y, device=device)
        probabilities = torch.tensor(clf.predict_proba(x), device=device)

        gold_label_probabilities = probabilities[torch.arange(probabilities.shape[0]), y.to(torch.int64)]

        true_probabilities = torch.max(probabilities, dim=1)[0]

        gold_label_probabilities = np.expand_dims(gold_label_probabilities.cpu().numpy(), axis=-1)
        true_probabilities = np.expand_dims(true_probabilities.cpu().numpy(), axis=-1)

        if self._gold_labels_probabilities is None:
            self._gold_labels_probabilities = gold_label_probabilities
        else:
            self._gold_labels_probabilities = np.hstack([self._gold_labels_probabilities, gold_label_probabilities])

        if self._true_probabilities is None:
            self._true_probabilities = true_probabilities
        else:
            self._true_probabilities = np.hstack([self._true_probabilities, true_probabilities])

    @property
    def gold_labels_probabilities(self):
        return self._gold_labels_probabilities

    @property
    def true_probabilities(self):
        return self._true_probabilities

    @property
    def confidence(self):
        return np.mean(self._gold_labels_probabilities, axis=-1)

    @property
    def variability(self):
        return np.std(self._gold_labels_probabilities, axis=-1)

    @property
    def correctness(self):
        return np.mean(self._gold_labels_probabilities > 0.5, axis=-1)
        # return np.mean(self._gold_labels_probabilities == self._true_probabilities, axis=-1)

    @property
    def aleatoric(self):
        preds = self._gold_labels_probabilities
        return np.mean(preds * (1 - preds), axis=-1)

# ===================== Main WorkFlow =====================

def is_better(current: dict, best: dict, metrics: list) -> bool:
    for key, mode in metrics:
        curr_v = current[key]
        best_v = best.get(key)
        
        if best_v is None:
            return True
        
        if curr_v == best_v:
            continue
        
        if mode == 'max' and curr_v > best_v:
            return True
        if mode == 'min' and curr_v < best_v:
            return True
        
        return False
    
    return False

def categorical_distance(generated_data, train_data, categorical_features):
    diff_matrix = np.zeros((generated_data.shape[0], train_data.shape[0]))
    for feature in categorical_features:
        gen_col = generated_data[feature].values[:, np.newaxis]
        train_col = train_data[feature].values[np.newaxis, :]
        diff_matrix += (gen_col != train_col).astype(int)
    return diff_matrix


def compute_dcr_and_entropy(train_data, generated_data, numerical_features, categorical_features):
    
    scaler = StandardScaler()
    train_num = scaler.fit_transform(train_data[numerical_features])
    gen_num = scaler.transform(generated_data[numerical_features])
    
    num_dist = cdist(gen_num, train_num, metric='cityblock')
    if categorical_features:
        cat_dist = categorical_distance(generated_data, train_data, categorical_features)
    else:
        cat_dist = 0
    total_dist = num_dist + cat_dist
    
    nearest_idx = np.argmin(total_dist, axis=1)
    
    x = train_data.shape[0]
    counts = np.bincount(nearest_idx, minlength=x)
    probs = counts / counts.sum()
    probs_nonzero = probs[probs > 0]
    entropy = -np.sum(probs_nonzero * np.log(probs_nonzero))
    return nearest_idx, counts, entropy


def entropy_ratio_with_cap(freqs, tau=0.75, r_max=0.6):
    freqs = np.array(freqs, dtype=float)
    p = freqs / freqs.sum() if freqs.sum()>0 else np.zeros_like(freqs)
    
    nz = p>0
    H = -np.sum(p[nz] * np.log(p[nz]))
    H_norm = H / np.log(len(freqs))
    return 1.0 - H_norm

def gini_ratio_with_cap(p):
    
    p = np.asarray(p, dtype=float)
    if p.sum() == 0:
        return 0.0
    p = p / p.sum()
    
    sorted_p = np.sort(p)
    n = len(sorted_p)
    indices = np.arange(1, n+1)
    return (1.0 / (n - 1)) * np.sum((2 * indices - n - 1) * sorted_p)

def filter_by_group_quality(
    X_small, y_small, X_large, y_large,
    numerical_features, categorical_features,
    samples_per_block=20,
    n_iterations=10,
    ratio2=0.75,
    curate = True   
):
    clf = XGBClassifier(n_estimators=100)
    clf.fit(X_small, y_small)
    train_df = X_small.copy()
    train_df['label'] = y_small
    gen_df = X_large.copy()
    gen_df['label'] = y_large

    categorical_features.append('label')

    nearest_idx, counts, _ = compute_dcr_and_entropy(
        train_data=train_df,
        generated_data=gen_df,
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )
    freqs = counts / counts.sum()

    order = np.argsort(-freqs)
    p_sorted = freqs[order]
    ratio = gini_ratio_with_cap(p_sorted)

    print("Proxy-Distribution:", p_sorted)

    cum_freqs = np.cumsum(freqs[order])
    t_star = np.searchsorted(cum_freqs, ratio) + 1

    ratio = np.sum(p_sorted[:t_star])
    print("ratio 1 =", ratio)

    selected_classes = order[:t_star]

    sel_mask = np.isin(nearest_idx, selected_classes)

    if t_star > 0:
        selected_classes = order[:t_star]
        sel_mask = np.isin(nearest_idx, selected_classes)
    else:
        sel_mask = np.zeros_like(nearest_idx, dtype=bool)
        print("t_star=0, all samples put into instance-level")

    print("high-frequency modes number:", t_star, "/", len(y_small), "high-frequency samples number:", np.sum(sel_mask), "+",len(y_large) - np.sum(sel_mask), "/", len(y_large))

    sel_indices = np.flatnonzero(sel_mask)
    other_indices = np.flatnonzero(~sel_mask)
    n_sel = len(sel_indices)

    from filtering import data_centric_curation
    final_nonred_idxs = np.array([], dtype=int)
    if curate and len(other_indices) > 0:
        X_other = X_large.iloc[other_indices]
        y_other = np.array(y_large)[other_indices]
        easy, ambig, _ = data_centric_curation(
            X_small, y_small,
            X_other, y_other,
            curation_metric='aleatoric',
            retrain=False,
            nest=100,
            ratio=ratio
        )
        final_nonred_idxs = other_indices[np.array(easy, dtype=int)]

    beta = len(final_nonred_idxs)/(len(y_large) - np.sum(sel_mask))
    print(f"ratio in low-frequency = {beta:.4f}")

    selected_redundant_idxs = np.array([], dtype=int)
    if n_sel > 0:
        X_sel = X_large.iloc[sel_indices].reset_index(drop=True)
        y_sel = np.array(y_large)[sel_indices]

        Filter_sel = Filter(X_sel, y_sel)
        for _ in range(n_iterations):
            Filter_sel.on_epoch_end(clf=clf)

        block_count = max(1, n_sel // samples_per_block)
        blocks = [list(range(i * samples_per_block,
                             min((i+1) * samples_per_block, n_sel)))
                  for i in range(block_count)]
        block_metrics = []
        for blk in blocks:
            Xi, yi = X_sel.iloc[blk], y_sel[blk]
            Filter_blk = Filter(Xi, yi)
            for _ in range(n_iterations):
                Filter_blk.on_epoch_end(clf=clf)
            block_metrics.append({'corr': np.mean(Filter_blk.correctness),
                                  'conf': np.mean(Filter_blk.confidence)})

        r_opt = 0.15 * np.log(ratio) + 0.55
        print(f"ratio 2 = {r_opt:.4f}")
        N_keep = int(np.floor(r_opt * n_sel)) + 1
        blocks_needed = int(np.ceil(N_keep / samples_per_block))

        sorted_blocks = sorted(
            enumerate(block_metrics),
            key=lambda x: (x[1]['corr'], x[1]['conf']),
            reverse=True
        )
        top_block_indices = [idx for idx, _ in sorted_blocks[:blocks_needed]]

        selected_redundant_idxs = np.concatenate([
            sel_indices[blocks[bi]] for bi in top_block_indices
        ]).astype(int)
        
    print("After filtering:", len(selected_redundant_idxs), "(", np.sum(sel_mask), ") +" , len(final_nonred_idxs), "(", len(y_large) - np.sum(sel_mask), ") /", len(y_large))

    final_indices = np.concatenate([selected_redundant_idxs, final_nonred_idxs]).astype(int)

    return final_indices,ratio,ratio2,beta


def grid_search_filter(
    X_small, y_small, X_large, y_large,
    X_test, y_test,
    numerical_features,
    categorical_features,
    block_sizes=[10,20,30],
    n_iterations=10,
    ratio2=0.75,
    curate = True
):
    best = {
        'score': -1, 
        'surprisal': None, 
        'entropy': None, 
        'ale': None,
        'block_sizes': 0,
        'ratio1': 0,
        'ratio2': 0,
        'beta': 0
    }

    for bs in block_sizes:
        idxs, ratio1, ratio2, beta = filter_by_group_quality(
            X_small,y_small, X_large, y_large, 
            numerical_features = numerical_features, 
            categorical_features = categorical_features,
            samples_per_block=bs,
            n_iterations=n_iterations,
            ratio2=ratio2,
            curate = curate
        )
        if len(idxs)==0: continue
        Xf=X_large.iloc[idxs]; yf=np.array(y_large)[idxs]
        clf2 = XGBClassifier(n_estimators=50)
        clf2.fit(Xf, yf)

        preds = clf2.predict(X_test)
        acc   = accuracy_score(preds, y_test)

        probs = clf2.predict_proba(X_test)

        gold_probs = probs[np.arange(len(y_test)), y_test.astype(int)]
        
        surprisal = -np.log(gold_probs + 1e-12)
        mean_surprisal = surprisal.mean()
        
        predictive_entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        mean_entropy = predictive_entropy.mean()
        
        aleatoric = gold_probs * (1 - gold_probs)
        mean_aleatoric = aleatoric.mean()

        current = {
            'score': acc,
            'surprisal': mean_surprisal,
            'entropy': mean_entropy,
            'ale': mean_aleatoric,
            'block_sizes': bs,
            'idxs':idxs,
            'ratio1': ratio1,
            'ratio2': ratio2,
            'beta': beta
        }
        
        metrics = [
            ('surprisal', 'min'),  
            ('score', 'max'),     
            ('entropy', 'min'),    
            ('ale', 'min'),         
        ]

        if is_better(current, best, metrics):
            best.update(current)

        print(f"bs={bs}, acc={acc:.3f}, surprisal={mean_surprisal:.3f}, kept={len(idxs)}")

    print("BEST: score", best['score'], "surprisal:", best['surprisal'], "block size:", best['block_sizes'], "idxs:", len(best['idxs']), "ratio1:", best['ratio1'], "beta:", best['beta'])
    return best
