from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import umap
from sdv.tabular import CTGAN, TVAE, GaussianCopula
import torch
import numpy as np
from imblearn.over_sampling import SMOTENC, SMOTE
from sdv.sampling import Condition

def generate_model(data, target, model, seed):

    data[target] = data[target].astype("str")

    final_data_gen = pd.DataFrame()


    if model == "ctgan":

        np.random.seed(seed)
        torch.manual_seed(seed)
        model_gen = CTGAN(cuda=True)

    elif model == "tvae":

        np.random.seed(seed)
        torch.manual_seed(seed)
        model_gen = TVAE(cuda=True)

    elif model == "copula":

        np.random.seed(seed)
        torch.manual_seed(seed)
        model_gen = GaussianCopula()

    model_gen.fit(
        data
    )

    for i in data[target].unique():
        
        condition = Condition({
            target: i}, num_rows=data[data[target] == i].shape[0])

        new_data_gen = model_gen.sample_conditions(conditions=[condition])
        new_data_gen = pd.DataFrame(new_data_gen)
        new_data_gen[target] = i

        final_data_gen = final_data_gen.append(new_data_gen)
        final_data_gen.reset_index(drop=True, inplace=True)
      
    final_data_gen[target] = final_data_gen[target].astype(int)
    data[target] = data[target].astype(int)
    
    return final_data_gen


def smotenc_model(
    data, num_cols, target, cat_cols=[],  number_neighs=5, seed=22,
):

    data[cat_cols] = data[cat_cols].astype(str)
    integers = []

    for i, j in zip(data[num_cols].dtypes, data[num_cols].dtypes.index):
        if i == int:
            integers.append(j)

    
    scaler = MinMaxScaler()

    scaler.fit(data[num_cols].copy())
    data_scaled = scaler.transform(data[num_cols].copy())
    data_scaled = pd.DataFrame(data_scaled, index=data.index, columns=num_cols)
    data_scaled = pd.concat([data_scaled, data[cat_cols]], axis=1)
    data_scaled[target] = data[target]

    synt_classes = {}

    for cla in data[target].unique():
        synt_classes[cla] = data[data[target] == cla].shape[0] * 2

    if len(cat_cols) != 0:
        cat_cols_pos = [
            data_scaled.columns.get_loc(col)
            for col in data_scaled.columns
            if col in cat_cols
        ]

        smote = SMOTENC(
            k_neighbors=number_neighs,
            sampling_strategy=synt_classes,
            categorical_features=cat_cols_pos,
            random_state=seed,
        )
    else:
        smote = SMOTE(
            k_neighbors=number_neighs,
            sampling_strategy=synt_classes,
            random_state=seed,
        )

    data_smotenc, y_resampled = smote.fit_resample(
        data_scaled[[col for col in data_scaled if col != target]], data_scaled[target]
    )

    data_smotenc = pd.DataFrame(data_smotenc)
    data_smotenc[target] = y_resampled
    data_smotenc = data_smotenc.iloc[-data_smotenc.shape[0] // 2 :, :]

    y_s = data_smotenc[target].copy()

    data_smotenc[num_cols] = pd.DataFrame(
        scaler.inverse_transform(data_smotenc[num_cols]),
        columns=num_cols,
        index=data_smotenc.index,
    )

    data_smotenc[target] = y_s.values

    for col in integers:

        data_smotenc[col] = data_smotenc[col].astype(int)

    return data_smotenc


def smotenc_umap_model(
    data,
    num_cols,
    target,
    cat_cols=[],
    use_umap=True,
    number_neighs=5,
    seed=22,
    min_dist=0.1,
    n_neighs = 15
):
    data[cat_cols] = data[cat_cols].astype(str)

    integers = []

    for i, j in zip(data[num_cols].dtypes, data[num_cols].dtypes.index):
        if i == int:
            integers.append(j)


    scaler = MinMaxScaler()

    scaler.fit(data[num_cols].copy())
    data_scaled = scaler.transform(data[num_cols].copy())
    data_scaled = pd.DataFrame(data_scaled, index=data.index, columns=num_cols)

    if use_umap == True:
        compressor = umap.UMAP(random_state=seed, n_components=2, min_dist = min_dist, n_neighbors= n_neighs)
        compressor.fit(data_scaled, y=data[target])
        data_scaled = compressor.transform(data_scaled)

        data_scaled = pd.DataFrame(data_scaled, index=data.index)
        data_scaled = pd.concat(
            [data_scaled, data.loc[data_scaled.index][cat_cols]], axis=1
        )


        data_scaled[target] = data.loc[data_scaled.index][target]

    synt_classes = {}
    for cla in data[target].unique():
        
        synt_classes[cla] = (data[data[target] == cla].shape[0]+ data_scaled[data_scaled[target] == cla].shape[0])

    if len(cat_cols) != 0:
        
        cat_cols_pos = [
            data_scaled.columns.get_loc(col)
            for col in data_scaled.columns
            if col in cat_cols
        ]

        smote = SMOTENC(
            k_neighbors=number_neighs,
            sampling_strategy=synt_classes,
            categorical_features=cat_cols_pos,
            random_state=seed,
        )
    else:
        smote = SMOTE(
            k_neighbors=number_neighs,
            sampling_strategy=synt_classes,
            random_state=seed,
        )
    data_smotenc, y_resampled = smote.fit_resample(
        data_scaled[[col for col in data_scaled if col != target]], data_scaled[target]
    )
    
    data_smotenc = pd.DataFrame(data_smotenc)
    data_smotenc[target] = y_resampled

    data_smotenc = data_smotenc.iloc[data_scaled.shape[0] :, :]
    y_s = data_smotenc[target].copy()

    data_smotenc.reset_index(drop=True, inplace=True)
    data_smotenc[num_cols] = pd.DataFrame(
        compressor.inverse_transform(data_smotenc[[0, 1]]),
        index=data.index,
        columns=num_cols,
    )

    data_smotenc[num_cols] = pd.DataFrame(
        scaler.inverse_transform(data_smotenc[num_cols]),
        columns=num_cols,
        index=data_smotenc.index,
    )

    data_smotenc.drop([0, 1], inplace=True, axis=1)

    data_smotenc[target] = y_s.values

    for col in integers:

        data_smotenc[col] = data_smotenc[col].astype(int)

    return data_smotenc



def binnarize(data, cat_cols):

    for i in cat_cols:
        bin = pd.get_dummies(data[i])
        bin_columns = [i + "_" + str(j) for j in bin.columns]
        bin.columns = bin_columns
        data = pd.concat([data, bin], axis=1)

    data = data.loc[:, [i for i in data.columns if i not in cat_cols]]

    return data