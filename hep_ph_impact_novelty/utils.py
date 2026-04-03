import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import torch
import torch.nn as nn


SUBFIELD_KEYWORDS = [
    ['LHC', 'ATLAS', 'CMS', 'Large Hadron Collider', 'ATLAS Collaboration', 'CMS Collaboration'],
    ['Dark Matter', 'DM', 'WIMP', 'WIMPs', 'Weakly Interacting Massive Particle', 'Dark Sector', 'Hidden Sector'],
    ['Axion', 'Axions', 'ALP', 'ALPs', 'QCD Axion', 'Axion-like particle', 'Axion-like particles'],
    ['Supersymmetry', 'SUSY', 'Supersymmetric', 'MSSM', 'NMSSM'],
    ['Composite Higgs', 'CHM', 'Little Higgs', 'Composite Higgs Model', 'Little Higgs Model'],
    ['Inflation', 'Inflaton', 'Cosmic inflation', 'Slow-roll','Slow roll'],
]


class ImpactFromFixedEmbedding(nn.Module):

    def __init__(self, embed_dim, n_layers=1, dim_hidden=64, device='cpu') -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.0], device=device))
        if n_layers == 1:
            self.layers = nn.Sequential(nn.Linear(embed_dim, 1, bias=True))
        else:
            layers = [nn.Linear(embed_dim, dim_hidden, bias=True), nn.ReLU()]
            layers += [nn.Sequential(nn.Linear(dim_hidden, dim_hidden, bias=True), nn.ReLU())
                       for _ in range(n_layers - 2)]
            layers += [nn.Linear(dim_hidden, 1, bias=True)]
            self.layers = nn.Sequential(*layers)

    def forward(self, x, y=None):
        logits = self.layers(x)
        loss = self.loss_fn(logits, y.float()) if y is not None else None
        return logits, loss


def prepare_split(features, labels, device, train_frac=0.8, val_frac=0.1):
    """
    Standardize features, convert to tensors, and split chronologically.
    Returns Xtr, Ytr, Xdev, Ydev, Xtest, Ytest.
    """
    means = features.mean(axis=0, keepdims=True)
    stds  = features.std(axis=0, keepdims=True)
    X = torch.tensor((features - means) / stds, dtype=torch.float32, device=device)
    Y = torch.tensor(labels, dtype=torch.float32, device=device).view(-1, 1)

    train_idx = int(train_frac * len(X))
    val_idx   = int((train_frac + val_frac) * len(X))

    Xtr,  Ytr  = X[:train_idx],        Y[:train_idx]
    Xdev, Ydev = X[train_idx:val_idx],  Y[train_idx:val_idx]
    Xtest,Ytest= X[val_idx:],           Y[val_idx:]

    print(f'Train: {len(Xtr):,} | Val: {len(Xdev):,} | Test: {len(Xtest):,}')
    return Xtr, Ytr, Xdev, Ydev, Xtest, Ytest,means,stds


def build_model(embed_dim, device, n_layers=1, dim_hidden=128, learning_rate=1e-4):
    """Instantiate model and optimizer, move to device."""
    model = ImpactFromFixedEmbedding(embed_dim=embed_dim, n_layers=n_layers,
                                      dim_hidden=dim_hidden, device=device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {n_layers} layer(s), {n_params} params, device={device}')
    return model, optimizer


def train_model(model, Xtr, Ytr, Xdev, Ydev, optimizer,
                max_iter=10000, batch_size=64, eval_interval=1000):
    for i in range(max_iter):
        idx_batch = np.random.randint(0, len(Xtr), size=batch_size)
        xb, yb = Xtr[idx_batch], Ytr[idx_batch]
        logits, loss = model(xb, yb)

        if i % eval_interval == 0:
            with torch.no_grad():
                _, train_loss = model(Xtr, Ytr)
                _, val_loss = model(Xdev, Ydev)
            print(f'{i}/{max_iter} : Train loss = {train_loss.item():.4f} | Val loss = {val_loss.item():.4f}')

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def evaluate_model(model, X, Y, interval=0.001):
    with torch.no_grad():
        logits, _ = model(X, None)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    truth_labels = Y.view(-1).cpu().numpy()
    thr_vals = np.arange(0.0, 1.0, interval)
    ROC, PR = [], []

    for thr in thr_vals:
        pred_labels = np.where(probs > thr, 1, 0)
        TP = ((pred_labels == 1) & (truth_labels == 1)).sum()
        FN = ((pred_labels == 0) & (truth_labels == 1)).sum()
        FP = ((pred_labels == 1) & (truth_labels == 0)).sum()
        TN = ((pred_labels == 0) & (truth_labels == 0)).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TPR

        ROC.append([FPR, TPR])
        PR.append([recall, precision])

    ROC = np.array(ROC)
    PR  = np.array(PR)
    ROC_AUC = ROC[:, 1].sum() * interval
    PR_AUC  = PR[:, 1].sum() * interval

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    base_rate = truth_labels.mean()

    axes[0].plot(ROC[:, 0], ROC[:, 1])
    axes[0].plot([0, 1], [0, 1], '--', color='gray', alpha=0.5, label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'ROC curve  (AUC = {ROC_AUC:.3f})')
    axes[0].legend()

    axes[1].scatter(PR[:, 0], PR[:, 1], s=2)
    axes[1].axhline(base_rate, linestyle='--', color='gray', alpha=0.5, label=f'Random ({base_rate:.2f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'PR curve  (AUC = {PR_AUC:.3f})')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return ROC_AUC, PR_AUC


def show_top_predictions(df, n=10, min_citation_rate=10):
    top = df.sort_values('probs', ascending=False).head(n)
    TP = (top['citation_rate'] > min_citation_rate).sum()
    for _, row in top.iterrows():
        title = row['title'][:70] + ('...' if len(row['title']) > 70 else '.')
        print(f"{row['citation_rate']:5.1f} cit/yr  |  {row['probs']:.2f}  |  {title}")
    print('-' * 100)
    print(f'Predicting {TP} out of {n} papers ({100*TP/n:.1f}%) with citation rate > {min_citation_rate} cit/yr')


def show_ahead_of_time_predictions(df, n=10, min_citation_rate=10):
    df_sorted = df.sort_values('probs').copy()
    df_sorted = df_sorted[df_sorted['citation_rate'] > min_citation_rate]
    top = df_sorted.head(n)
    for _, row in top.iterrows():
        title = row['title'][:70] + ('...' if len(row['title']) > 70 else '.')
        print(f"{row['citation_rate']:5.1f} cit/yr  |  {row['probs']:.2f}  |  {title}")


def plot_citation_rates(df, keywords=[], last_year=2025):
    pattern = '|'.join(map(re.escape, keywords))
    mask = df['abstract'].str.contains(pattern, case=False, na=False)
    df_subset = df.loc[mask].copy()

    df_subset['d_year'] = pd.Timestamp.now().year - df_subset['year']
    df_subset['citation_rate'] = df_subset['citation_count'] / df_subset['d_year']

    years = df_subset['year'].value_counts().sort_index().index
    year_mask = years <= last_year

    rate_med = df_subset.groupby('year')['citation_rate'].median()
    quantiles_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    quantiles = [df_subset.groupby('year')['citation_rate'].quantile(q) for q in quantiles_vals]

    plt.figure(figsize=(14, 6))
    for i, quant in enumerate(quantiles):
        plt.plot(
            years[year_mask], quant[year_mask],
            label=f'{quantiles_vals[i]}',
            color='purple',
            alpha=1.0 - 2.2 * abs(quantiles_vals[i] - 0.5),
            marker='o' if quantiles_vals[i] == 0.5 else None
        )
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Citation rate", fontsize=14)
    plt.title(f"Citation rate by publication year, category = {keywords[0] if keywords else 'all'}", fontsize=16)
    plt.xticks(years[year_mask], rotation=45)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_citation_subfields(df, subfields, last_year=2025):
    patterns = ['|'.join(map(re.escape, keywords)) for keywords in subfields]
    masks = [df['abstract'].str.contains(p, case=False, na=False) for p in patterns]
    df_subsets = [df.loc[mask].copy() for mask in masks]

    for subset in df_subsets:
        subset['d_year'] = pd.Timestamp.now().year - subset['year']
        subset['citation_rate'] = subset['citation_count'] / subset['d_year']

    df = df.copy()
    df['d_year'] = pd.Timestamp.now().year - df['year']
    df['citation_rate'] = df['citation_count'] / df['d_year']

    years = df['year'].value_counts().sort_index().index
    year_mask = years <= last_year

    colors = sns.color_palette('husl', len(subfields))
    medians = [subset.groupby('year')['citation_rate'].median() for subset in df_subsets]
    field_wide_median = df.groupby('year')['citation_rate'].median()

    plt.figure(figsize=(14, 6))
    for i, median in enumerate(medians):
        plt.plot(
            years[year_mask], median.iloc[year_mask],
            label=subfields[i][0], color=colors[i], linewidth=2, marker='o', markersize=6
        )
    plt.plot(
        years[year_mask], field_wide_median.iloc[year_mask],
        label='Field-wide', color='black', alpha=0.3, linewidth=3, linestyle='--'
    )
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Median citation rate", fontsize=14)
    plt.title("Citation rate medians for different subfields", fontsize=16)
    plt.xticks(years[year_mask], rotation=45)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_subfields(df, embedding_transformed, subfields):
    patterns = ['|'.join(map(re.escape, keywords)) for keywords in subfields]
    masks = [df['abstract'].str.contains(p, case=False, na=False) for p in patterns]

    n = len(subfields)
    cols = 3
    rows = (n + cols - 1) // cols
    colors = sns.color_palette('husl', n)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    for i, mask in enumerate(masks):
        ax = axes[i]
        ax.scatter(embedding_transformed[:, 0], embedding_transformed[:, 1],
                   color='lightgray', s=3, alpha=0.3)
        ax.scatter(embedding_transformed[mask.values, 0], embedding_transformed[mask.values, 1],
                   color=colors[i], s=7)
        ax.set_title(subfields[i][0])
        ax.set_xticks([])
        ax.set_yticks([])
        pct = 100 * mask.sum() / len(df)
        ax.text(0.97, 0.97, f'{pct:.1f}%', transform=ax.transAxes,
                ha='right', va='top', fontsize=10, color=colors[i])

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_umap(df, embedding_transformed):
    t = df['earliest_date'].astype('int64')
    t_norm = (t - t.min()) / (t.max() - t.min())

    dates = pd.to_datetime(df['earliest_date'])
    age_years = (pd.Timestamp.now() - dates).dt.days / 365.25
    log_rate = np.log10(1 + df['citation_count'] / (age_years + 1e-3))
    vmax = np.percentile(log_rate, 95)

    order = np.random.permutation(len(embedding_transformed))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc0 = axes[0].scatter(
        embedding_transformed[order, 0], embedding_transformed[order, 1],
        c=t_norm.iloc[order], cmap='viridis', s=2, alpha=1.0
    )
    cbar0 = plt.colorbar(sc0, ax=axes[0], label='Year')
    ticks = np.linspace(0, 1, 5)
    tick_dates = pd.to_datetime(ticks * (t.max() - t.min()) + t.min())
    cbar0.set_ticks(ticks)
    cbar0.set_ticklabels(tick_dates.strftime('%Y'))
    axes[0].set_title('Colored by publication year')

    sc1 = axes[1].scatter(
        embedding_transformed[order, 0], embedding_transformed[order, 1],
        c=log_rate.iloc[order], cmap='viridis', vmin=log_rate.min(), vmax=vmax, s=2
    )
    plt.colorbar(sc1, ax=axes[1], label='log(1 + citations/year)')
    axes[1].set_title('Colored by citation rate')

    for ax in axes:
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('SPECTER2 embedding space (UMAP projection)', y=1.02)
    plt.tight_layout()
    plt.show()


def build_subset(df_ordered, embeddings_ordered, keywords=[], window_end=6, window_length=12):
    pattern = '|'.join(map(re.escape, keywords))
    mask = df_ordered['abstract'].str.contains(pattern, case=False, na=False)

    df_subset = df_ordered[mask]
    embeddings_subset = embeddings_ordered[mask]

    dates = pd.to_datetime(df_subset['earliest_date']).to_numpy()
    interval_start = (df_subset['earliest_date'] - pd.DateOffset(months=window_end + window_length)).to_numpy()
    interval_end   = (df_subset['earliest_date'] - pd.DateOffset(months=window_end)).to_numpy()

    left_idx  = np.searchsorted(dates, interval_start, side='left')
    right_idx = np.searchsorted(dates, interval_end,   side='right')

    E = embeddings_subset
    cumsum = np.cumsum(E, axis=0)
    cumsum = np.vstack([np.zeros((1, E.shape[1]), dtype=E.dtype), cumsum])

    lengths = (right_idx - left_idx)[:, None]
    sums = cumsum[right_idx] - cumsum[left_idx]

    clip_id = np.flatnonzero(lengths > 100)[0]

    embed_past_means  = sums[clip_id:] / lengths[clip_id:]
    df_clipped        = df_subset.iloc[clip_id:].copy()
    embeddings_clipped = embeddings_subset[clip_id:]

    embed_past_means_normalized   = embed_past_means / np.linalg.norm(embed_past_means, axis=1, keepdims=True)
    embeddings_clipped_normalized = embeddings_clipped / np.linalg.norm(embeddings_clipped, axis=1, keepdims=True)

    novelty_score = 0.5 * (1 - np.sum(embed_past_means_normalized * embeddings_clipped_normalized, axis=1))
    df_clipped['novelty_score'] = novelty_score

    dates = pd.to_datetime(df_clipped['earliest_date'])
    age_years = (pd.Timestamp.now() - dates).dt.days / 365.25
    df_clipped['citation_rate'] = df_clipped['citation_count'] / (age_years + 1e-3)
    df_clipped['log_rate'] = np.log10(1 + df_clipped['citation_rate'])

    return df_clipped, embeddings_clipped



def plot_novelty_over_time_combined(df_field, subfield_data, labels, yrange=[0.02, 0.06]):
    """
    Overlay novelty-over-time curves for the full field and multiple subfields.

    Args:
        df_field      : field-wide df with novelty_score (from compute_novelty_subset)
        subfield_data : list of (df_clipped, _) tuples, one per subfield
        labels        : list of subfield label strings
        yrange        : [ymin, ymax] for the y-axis
    """
    colors = sns.color_palette('husl', len(labels))

    def smooth_series(df):
        data = df.copy()
        data['earliest_date'] = pd.to_datetime(data['earliest_date'])
        data = data.set_index('earliest_date').sort_index()
        y = data['novelty_score']
        y_mean = y.resample('ME').mean()
        y_std  = y.resample('ME').std()
        return (y_mean.rolling(6, center=True, min_periods=1).mean(),
                y_std.rolling(6, center=True, min_periods=1).mean())

    plt.figure(figsize=(12, 6))

    mean, std = smooth_series(df_field)
    plt.plot(mean, label='Field-wide', color='black', linewidth=2, linestyle='--')

    for (df_sub, _), label, color in zip(subfield_data, labels, colors):
        mean, _ = smooth_series(df_sub)
        plt.plot(mean, label=label, color=color, linewidth=1.5)

    plt.xlabel('Date')
    plt.ylabel('Novelty score')
    plt.title('Novelty over time by subfield')
    plt.ylim(yrange[0], yrange[1])
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_uzzi_combined(df_field, subfield_data, labels, q=10):
    """
    Novelty-impact subplots, one per subfield, with the field-wide curve as reference.

    Args:
        df_field      : field-wide df with novelty_score
        subfield_data : list of (df_clipped, _) tuples
        labels        : list of subfield label strings
        q             : number of novelty bins
    """
    colors = sns.color_palette('husl', len(labels))

    def compute_curve(df):
        df = df.copy()
        df['novelty_score_bins'] = pd.qcut(df['novelty_score'], q=q)
        mean    = df.groupby('novelty_score_bins', observed=False)['log_rate'].mean()
        std     = df.groupby('novelty_score_bins', observed=False)['log_rate'].std()
        n       = df.groupby('novelty_score_bins', observed=False).size()
        centers = mean.index.map(lambda x: x.mid).values
        xerr    = np.array([iv.length for iv in mean.index]) / 2.0
        yerr    = std.values / np.sqrt(n.values)
        return centers, mean.values, xerr, yerr

    field_centers, field_mean, field_xerr, field_yerr = compute_curve(df_field)

    n = len(labels)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=False)
    axes = axes.flatten()

    for i, ((df_sub, _), label, color) in enumerate(zip(subfield_data, labels, colors)):
        ax = axes[i]
        centers, mean, xerr, yerr = compute_curve(df_sub)

        # field-wide reference
        ax.errorbar(field_centers, field_mean, yerr=field_yerr, xerr=field_xerr,
                    fmt='o--', capsize=2, color='black', alpha=0.4, linewidth=1, label='Field-wide')
        # subfield
        ax.errorbar(centers, mean, yerr=yerr, xerr=xerr,
                    fmt='o-', capsize=2, color=color, linewidth=1.5, label=label)

        ax.set_title(label)
        ax.set_xlabel('Novelty score')
        ax.set_ylabel('Mean log citation rate')
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Novelty vs. impact by subfield', y=1.02)
    plt.tight_layout()
    plt.show()


