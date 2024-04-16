import pandas as pd
from umap import UMAP
from imblearn.over_sampling import SMOTENC, SMOTE
from sklearn.preprocessing import MinMaxScaler

class UMAPSMOTENC:
    
    def __init__(
        self,
        num_cols,
        target,
        cat_cols=[],
        use_umap=True,
        number_neighs=5,
        seed=22,
        min_dist=0.1,
        n_neighs=15
    ):
        self.num_cols = num_cols
        self.target = target
        self.cat_cols = cat_cols
        self.use_umap = use_umap
        self.number_neighs = number_neighs
        self.seed = seed
        self.min_dist = min_dist
        self.n_neighs = n_neighs
        self.scaler = MinMaxScaler()
        self.compressor = UMAP(
            random_state=self.seed,
            n_components=2,
            min_dist=self.min_dist,
            n_neighbors=self.n_neighs
        )

    def fit_transform(self, data):
        
        data[self.cat_cols] = data[self.cat_cols].astype(str)

        integers = [col for col, dtype in zip(data[self.num_cols].dtypes.index, data[self.num_cols].dtypes) if dtype == int]

        self.scaler.fit(data[self.num_cols].copy())
        data_scaled = self.scaler.transform(data[self.num_cols].copy())
        data_scaled = pd.DataFrame(data_scaled, index=data.index, columns=self.num_cols)

        if self.use_umap:
            self.compressor.fit(data_scaled, y=data[self.target])
            data_scaled = self.compressor.transform(data_scaled)

            data_scaled = pd.DataFrame(data_scaled, index=data.index)
            data_scaled = pd.concat(
                [data_scaled, data.loc[data_scaled.index][self.cat_cols]], axis=1
            )
            data_scaled[self.target] = data.loc[data_scaled.index][self.target]

        synt_classes = {cla: (data[data[self.target] == cla].shape[0] + data_scaled[data_scaled[self.target] == cla].shape[0]) for cla in data[self.target].unique()}

        if len(self.cat_cols) != 0:
            cat_cols_pos = [data_scaled.columns.get_loc(col) for col in data_scaled.columns if col in self.cat_cols]
            smote = SMOTENC(
                k_neighbors=self.number_neighs,
                sampling_strategy=synt_classes,
                categorical_features=cat_cols_pos,
                random_state=self.seed,
            )
        else:
            smote = SMOTE(
                k_neighbors=self.number_neighs,
                sampling_strategy=synt_classes,
                random_state=self.seed,
            )
        
        data_scaled.columns = data_scaled.columns.astype(str)
        data_smotenc, y_resampled = smote.fit_resample(
            data_scaled[[col for col in data_scaled if col != self.target]], data_scaled[self.target]
        )
        
        data_smotenc = pd.DataFrame(data_smotenc)
        data_smotenc[self.target] = y_resampled
        data_smotenc = data_smotenc.iloc[data_scaled.shape[0]:, :]
        y_s = data_smotenc[self.target].copy()
        data_smotenc.reset_index(drop=True, inplace=True)
        data_smotenc[self.num_cols] = pd.DataFrame(
            self.compressor.inverse_transform(data_smotenc[["0", "1"]]),
            columns=self.num_cols,
        )
        
        data_smotenc[self.num_cols] = pd.DataFrame(
            self.scaler.inverse_transform(data_smotenc[self.num_cols]),
            columns=self.num_cols,
            index=data_smotenc.index,
        )

        data_smotenc.drop(["0", "1"], inplace=True, axis=1)
        data_smotenc[self.target] = y_s.values
        
        for col in integers:
            data_smotenc[col] = data_smotenc[col].astype(int)

        return data_smotenc