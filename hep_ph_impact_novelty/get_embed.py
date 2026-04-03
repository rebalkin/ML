import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

DATA_DIR = Path('data')

tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

df = pd.read_parquet(DATA_DIR / 'papers.parquet')
text_batch = (df['title'] + tokenizer.sep_token + df['abstract']).dropna().tolist()

batch_size = 128
input_batches = []
for i in range(0, len(text_batch), batch_size):
    batch_input = tokenizer(
        text_batch[i:i+batch_size],
        padding=True, truncation=True,
        return_tensors="pt", return_token_type_ids=False, max_length=512
    )
    input_batches.append(batch_input)

all_embeddings = []
with torch.no_grad():
    for i, batch_input in enumerate(input_batches):
        inputs = {k: v.to(device) for k, v in batch_input.items()}
        output = model(**inputs)
        embeddings_batch = output.last_hidden_state[:, 0, :]
        all_embeddings.append(embeddings_batch.detach().cpu().numpy())
        if i % 100 == 0:
            print(f'Batch {i}/{len(input_batches)} done.')

embeddings = np.vstack(all_embeddings)
np.save(DATA_DIR / 'embeddings.npy', embeddings)