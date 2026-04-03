"""
Run once to generate umap_animation.html.
Requires data/papers.parquet and data/embedding_transformed.npy to be present.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

embedding_transformed = np.load('data/embedding_transformed.npy')

df = pd.read_parquet('data/papers.parquet')
df = df[df['abstract'].notna()]
df['earliest_date'] = pd.to_datetime(df['earliest_date'], format='mixed')

dates = pd.to_datetime(df['earliest_date'])
x = embedding_transformed[:, 0]
y = embedding_transformed[:, 1]

cmap = plt.cm.viridis
t = dates.astype('int64').to_numpy()
t_norm = (t - t.min()) / (t.max() - t.min())
base_rgba = cmap(t_norm)

fade_days = 365
frames = pd.date_range('2000-01', '2025-12', freq='MS')
dates_np = dates.to_numpy().astype('datetime64[D]')

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.set_xlim(x.min() - 1, x.max() + 1)
ax.set_ylim(y.min() - 1, y.max() + 1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('UMAP 1', color='white')
ax.set_ylabel('UMAP 2', color='white')
ax.set_title('SPECTER2 embedding space (2000-2025)', color='white')

scat = ax.scatter([], [], s=2)
date_label = ax.text(0.97, 0.97, '', transform=ax.transAxes,
                     fontsize=14, ha='right', va='top', fontweight='bold', color='white')

def update(frame_date):
    frame_np = np.datetime64(frame_date.date(), 'D')
    mask = dates_np <= frame_np
    colors = base_rgba[mask].copy()
    age_days = (frame_np - dates_np[mask]) / np.timedelta64(1, 'D')
    colors[:, 3] = np.clip(1.0 - age_days / fade_days, 0.05, 1.0)
    scat.set_offsets(np.column_stack([x[mask], y[mask]]))
    scat.set_color(colors)
    date_label.set_text(frame_date.strftime('%Y-%m'))
    return scat, date_label

plt.rcParams['animation.embed_limit'] = 50
anim = FuncAnimation(fig, update, frames=frames, interval=80, blit=True)
plt.close()

with open('umap_animation.html', 'w') as f:
    f.write(anim.to_jshtml())

print('Saved umap_animation.html')
