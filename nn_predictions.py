import torch
from torch import nn
import cudf
import numpy as np

DEVICE = 'cuda'


class Dataset:
    
    def __init__(self, df, device):
        self._cam_ids = torch.tensor(df['cam_id'].values, device=device)
        self._banner_ids = torch.tensor(df['banner_code'].values, device=device, dtype=torch.int64)

    def __getitem__(self, indices: int):
        return self._cam_ids[indices], self._banner_ids[indices]

    def get_all_data(self):
        return self._cam_ids, self._banner_ids


def get_low_high(df, device, num_users):
    low = {}
    high = {}
    prev_u_id = -1
    for i, u_id in enumerate(df['u_id'].to_pandas()):
        if prev_u_id != u_id:
            high[prev_u_id] = i
            low[u_id] = i
        prev_u_id = u_id
    high[prev_u_id] = len(df)
    low_tensor = torch.tensor([low.get(i, -1) for i in range(num_users)], device=device)
    high_tensor = torch.tensor([high.get(i, -1) for i in range(num_users)], device=device)
    return low_tensor, high_tensor

def randint_different_ranges(low, high, device):
    ranges = high - low
    rand_floats = torch.rand(low.shape, device=device)
    rand_scaled = (rand_floats * ranges).floor().long()
    return rand_scaled + low

class CommonDataset:
    
    def __init__(self, df, device, num_users):
        pos_df = df[df['target'] == 1]
        neg_df = df[df['target'] == 0]
        self._pos_dataset = Dataset(pos_df, device=device)
        self._neg_dataset = Dataset(neg_df, device=device)

        self._u_ids = torch.tensor(df['u_id'].unique(), device=device)
        self._pos_low, self._pos_high = get_low_high(pos_df, device=device, num_users=num_users)
        self._neg_low, self._neg_high = get_low_high(neg_df, device=device, num_users=num_users)
        self._device = device

    def get_batch(self):
        pos_index = randint_different_ranges(
            self._pos_low[self._u_ids], 
            self._pos_high[self._u_ids], 
            device=self._device,
        )
        neg_index = randint_different_ranges(
            self._neg_low[self._u_ids],
            self._neg_high[self._u_ids],
            device=self._device,
        )
        return self._pos_dataset[pos_index], self._neg_dataset[neg_index]

class Model(nn.Module):
    
    def __init__(self, num_campaigns, num_banners, device):
        
        super().__init__()

        self._num_banners = num_banners
        self._campaign_embeddings = nn.Embedding(
            num_embeddings=num_campaigns,
            embedding_dim=8,
            device=device,
        )
        self._linear1 = nn.Linear(
            in_features=8 + num_banners,
            out_features=128,
            device=device,
        )
        self._linear2 = nn.Linear(
            in_features=128,
            out_features=1,
            device=device,
        )
        self._relu = nn.ReLU()
            
        nn.init.xavier_uniform_(self._campaign_embeddings.weight)
        nn.init.xavier_uniform_(self._linear1.weight)
        nn.init.xavier_uniform_(self._linear2.weight)
        nn.init.zeros_(self._linear1.bias)
        nn.init.zeros_(self._linear2.bias)

    def forward(self, batch):
        cam_ids, banner_ids = batch
        campaign_embeddings = self._campaign_embeddings(cam_ids)
        banner_one_hot = nn.functional.one_hot(banner_ids, num_classes=self._num_banners).float()
        x = torch.cat([
            campaign_embeddings,
            banner_one_hot, 
        ], axis=1)
        x = self._linear1(x)
        x = self._relu(x)
        x = self._linear2(x)
        return x


def create_encoder(ser):
    values = ser.unique()
    return dict(zip(values.to_pandas().values, range(len(values))))
    
def get_users_mask(data):
    user_num_targets = data.groupby('user_id')['target'].nunique()
    return data['user_id'].isin(
        user_num_targets[user_num_targets == 2].index
    )
    
def BPR_loss(pos_logit, neg_logit):
    return -torch.log(torch.sigmoid(pos_logit - neg_logit)).mean()
    
def get_nn_predictions(train, test):

    train = train[get_users_mask(train)]
    train = train.sort_values(['user_id', 'adv_campaign_id'])

    user_encoder = create_encoder(cudf.concat([train['user_id'], test['user_id']])) 
    adv_campaign_encoder = create_encoder(cudf.concat([train['adv_campaign_id'], test['adv_campaign_id']]))

    train['u_id'] = train['user_id'].map(user_encoder)
    train['cam_id'] = train['adv_campaign_id'].map(adv_campaign_encoder)
    
    test['u_id'] = test['user_id'].map(user_encoder)
    test['cam_id'] = test['adv_campaign_id'].map(adv_campaign_encoder)
    
    train_dataset = CommonDataset(train, device=DEVICE, num_users=len(user_encoder))
    test_dataset = Dataset(test, device=DEVICE)

    model = Model(
        num_campaigns=len(adv_campaign_encoder),
        num_banners=train['banner_code'].max() + 1,
        device=DEVICE,
    )
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0005)

    for epoch in range(500):
        pos_batch, neg_batch = train_dataset.get_batch()
        optimizer.zero_grad()
        pos_logit = model(pos_batch)
        neg_logit = model(neg_batch)
        logit = torch.cat([pos_logit, neg_logit], axis=1)
        loss = BPR_loss(pos_logit, neg_logit)
        loss.backward()
        optimizer.step()
    

    return model(test_dataset.get_all_data()).flatten().detach().cpu().numpy()

def get_nn_predictions_mean(train, test, num_iterations=16):
    return np.mean([
        get_nn_predictions(train, test) for _ in range(num_iterations)
    ], axis=0)
