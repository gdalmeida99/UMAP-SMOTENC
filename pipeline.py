import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from generative_functions import (
    smotenc_umap_model, 
    generate_model,
    binnarize,
    smotenc_model
)
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from time import time
from sdv.evaluation import evaluate
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
import phik
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OrdinalEncoder
import torch
torch.cuda.empty_cache()

#Gridsearch Hyperparameters
########################
param_grids_lr = [{"solver": ["liblinear", "lbfgs"], "max_iter": [200, 500, 1500]}]

param_grids_rf = [{"n_estimators": [150, 250], "max_depth": [15, 25, None]}]
param_grids_dt = [{"max_depth": [15, 25, None]}]

param_grids_mlp = [
    {
        "learning_rate": ["constant", "adaptative"],
        "hidden_layer_sizes": [(100,), (100, 100), (200,), (150, 100),],
        "max_iter": [200, 500, 1500],
    }
]

param_grids_gs = [{"var_smoothing":[1e-9]}] #This constitutes a default hyperparameters for Naive Bayes.

param_grids_xgb = [
    {"max_depth": [6, 10, 15], "n_estimators": [100, 150, 250],},
]
########################

#Dataset Pre-Processing
########################
for iter_dataset in tqdm(["adult_st", "creditcard_st","adult_st", "occupancy_st", "beans_st","magic_st","digits_st", "loan_st","covertype_st"]):  
    
    storage = pd.DataFrame()
    num_flag=True

 
    if iter_dataset == "adult_st":
        
        num_flag=False
        data = pd.read_csv("adult.csv")
        data.income = data.income.apply(lambda x: 1 if x == ">50K" else 0)
        target = "income"
        num_cols = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"]
        cat_cols = [
            "workclass",
            "education",
            "educational-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "gender",
            "native-country",
        ]
        data[cat_cols] = data[cat_cols].astype(str)
    
    elif iter_dataset == "covertype_st":
        num_flag=False
        data = pd.read_csv("covtype.csv")

        _, data = train_test_split(data, test_size=25000, stratify=data.Cover_Type)

        data.reset_index(drop=True, inplace=True)
        soils = [col for col in data.columns if "Soil" in col]

        wild = [col for col in data.columns if "Wilderness" in col]

        data[wild] = data[wild].replace(
            1, pd.Series(data[wild].columns, data[wild].columns)
        )
        data[wild] = data[wild].applymap(lambda x: np.nan if x == 0 else x)
        data[wild] = data[wild].ffill(axis=1).bfill(axis=1)

        data[soils] = data[soils].replace(
            1, pd.Series(data[soils].columns, data[soils].columns)
        )
        data[soils] = data[soils].applymap(lambda x: np.nan if x == 0 else x)
        data[soils] = data[soils].ffill(axis=1).bfill(axis=1)

        data["Wilderness"] = data[wild[0]]
        data["Soil_Type"] = data[soils[0]]
        data = data[
            [col for col in data.columns if (col not in soils) and (col not in wild)]
        ]
        num_cols = [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
        ]
        cat_cols = ["Wilderness", "Soil_Type"]

        target = "Cover_Type"

        data[cat_cols] = data[cat_cols].astype(str)
        data[target] = data[target].apply(lambda x: x-1) 


    elif iter_dataset == "digits_st":
        

        data = pd.read_csv("letter-recognition.data", header = None)
        data.columns = data.columns.astype(str)
        target = "0"
        cat_cols = []
        enc = OrdinalEncoder()
        data[target] = enc.fit_transform(np.array(data[target] ).reshape(-1,1))
        data[target] = data[target].astype(int)
        num_cols = [col for col in data.columns if col != target]
        data[cat_cols] = data[cat_cols].astype(str)

        
    elif iter_dataset == "magic_st":
        
        data = pd.read_csv("magic04.data", header = None)
        data.columns = data.columns.astype(str)
        target = "10"
        cat_cols = []
        num_cols = [col for col in data.columns if col != target]
        data[target] = data[target].apply(lambda x: 0 if x=="h" else 1)
        data[cat_cols] = data[cat_cols].astype(str)

    elif iter_dataset == "loan_st":
        
        num_flag=False
        data = pd.read_csv("loan.csv")
        data.drop("ID", axis=1, inplace=True)
        target = "Personal Loan"
        cat_cols = [
            "ZIP Code",
            "Education",
            "Securities Account",
            "CD Account",
            "Online",
            "CreditCard",
        ]
        num_cols = ["Age", "Experience", "Income", "Family", "CCAvg", "Mortgage"]
        data[cat_cols] = data[cat_cols].astype(str)


    elif iter_dataset == "occupancy_st":

        data = pd.read_csv("occupancy1.txt")
        data1 = pd.read_csv("occupancy2.txt")
        data2 = pd.read_csv("occupancy3.txt")
        data = data.append(data1).append(data2)
        data.reset_index(drop=True, inplace=True)
        data.drop("date", axis=1, inplace=True)
        target = "Occupancy"
        num_cols = data.columns
        cat_cols = []
        num_cols = num_cols.drop("Occupancy")
        data.reset_index(drop=True, inplace=True)

    elif iter_dataset == "beans_st":
        
        data = pd.read_csv("Dry_Bean.csv")
        target = "Class"
        num_cols =[col for col in data.columns if col != target]
        cat_cols =[]
        enc = OrdinalEncoder()
        data.Class = enc.fit_transform(np.array(data.Class).reshape(-1,1)).astype(int)

    else:

        data = pd.read_csv("creditcard.csv")
        cat_cols = []
        num_cols = [col for col in data.columns if col != "Class"]
        target = "Class"

########################

    seeds = [220, 320, 420]

    for seed in tqdm(seeds, leave=False):

        counter = 0

        skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

        for train_index, test_index in tqdm(
            skf.split(
                data.loc[:, [col for col in data.columns if col != target]],
                data[target],
            ),
            leave=False,
        ):
            time1 = time.time()

            train, test = (
                data.loc[train_index, :].copy(),
                data.loc[test_index, :].copy(),
            )

            train.reset_index(inplace=True, drop=True)
            test.reset_index(inplace=True, drop=True)
            
            data_copula = generate_model(train.copy(), target, "copula", seed)
            bin_data_copula = binnarize(data_copula.copy(), cat_cols)
            data_copula.to_csv("data_copula_"+str(iter_dataset)+"_"+str(seed)+"_"+ str(counter)+".csv")
                
            data_vae = generate_model(train.copy(), target, "tvae", seed)
            bin_data_vae = binnarize(data_vae.copy(), cat_cols)
            data_vae.to_csv("data_vae_"+str(iter_dataset)+"_"+str(seed)+"_"+ str(counter)+".csv")
            
            data_gan = generate_model(train.copy(), target, "ctgan", seed)
            bin_data_gan = binnarize(data_gan.copy(), cat_cols)
            data_gan.to_csv("data_gan_"+str(iter_dataset)+"_"+str(seed)+"_"+ str(counter)+".csv")
          
            data_umap_smote_nc = smotenc_umap_model(
                train.copy(),
                num_cols.copy(),
                target,
                cat_cols.copy(),
                use_umap=True,
                number_neighs=5,
                seed=seed,
            )

            for tipo, coluna in zip(train.dtypes, train.dtypes.index):
                data_umap_smote_nc[coluna] = data_umap_smote_nc[coluna].astype(tipo)

            bin_data_umap_smote_nc = binnarize(data_umap_smote_nc.copy(), cat_cols)
            data_umap_smote_nc.to_csv("data_umap_smote_nc_"+str(iter_dataset)+"_"+ str(seed)+"_"+ str(counter)+".csv")
            
            data_smote_nc = smotenc_model(
                train.copy(),
                num_cols.copy(),
                target,
                cat_cols.copy(),
                number_neighs=5,
                seed=seed,
            )

            for tipo, coluna in zip(train.dtypes, train.dtypes.index):
                data_smote_nc[coluna] = data_smote_nc[coluna].astype(tipo)

            bin_data_smote_nc = binnarize(data_smote_nc.copy(), cat_cols)
            data_smote_nc.to_csv("data_smote_nc_"+str(iter_dataset)+"_"+str(seed)+"_"+ str(counter)+".csv")
        
            train.to_csv("train_"+str(iter_dataset)+"_"+str(seed)+"_"+ str(counter)+".csv")
            bin_train = binnarize(train.copy(), cat_cols)
            bin_test = binnarize(test.copy(), cat_cols)
            
            for dataset, bin_dataset, title in tqdm(zip(
                    [
                        data_smote_nc,
                        data_umap_smote_nc,
                        data_gan,
                        data_vae,
                        data_copula,
                        train,
        
                    ],
                    [
                        bin_data_smote_nc,
                        bin_data_umap_smote_nc,
                        bin_data_gan,
                        bin_data_vae,
                        bin_data_copula,
                        bin_train,
                
                    ],
                    [
                        "smote_nc",
                        "umap_smote_nc",
                        "gan",
                        "vae",
                        "copula",
                        "train",
                    ],
                ),
                leave=False,
            ):
                
                
                temp = pd.DataFrame()
                testar_adapted = bin_test.copy()
                testar_adapted[
                    [
                        col
                        for col in bin_dataset.columns
                        if col not in testar_adapted.columns
                    ]
                ] = 0

                testar_adapted = testar_adapted[bin_dataset.columns]
                
                scaler = MinMaxScaler()
                scaler.fit(bin_dataset[num_cols])

                aux_dataset_num_train = pd.DataFrame(scaler.transform(bin_dataset[num_cols]), index = bin_dataset.index, columns = num_cols)
                aux_dataset_num_test = pd.DataFrame(scaler.transform(testar_adapted[num_cols]), index = testar_adapted.index, columns = num_cols)

                aux_dataset_num_train[[col for col in bin_dataset.columns if col not in aux_dataset_num_train.columns]] = bin_dataset[[col for col in bin_dataset.columns if col not in aux_dataset_num_train.columns]]
                aux_dataset_num_test[[col for col in testar_adapted.columns if col not in aux_dataset_num_test.columns]] = testar_adapted[[col for col in testar_adapted.columns if col not in aux_dataset_num_test.columns]]

                X = aux_dataset_num_train.append(aux_dataset_num_test).reset_index(drop=True)
                y = X[target]
                X = X.loc[:, [i for i in bin_dataset.columns if i != target]]
                
                #fnlwgt constitutes an important variable for synthetic data diversity. Neverthless, for classification performance it leads to suboptimal perfomance across all techiques.
                if iter_dataset == "adult_st":
                    X.drop("fnlwgt", axis=1, inplace=True)

                separation = np.where(
                    X.index <= bin_dataset.reset_index(drop=True).index.max(), -1, 0
                )

                ps = PredefinedSplit(separation)

                
                model1 = LogisticRegression(random_state=seed, n_jobs=-1)
                model2 = RandomForestClassifier(n_jobs=-1, random_state=seed)
                model3 = DecisionTreeClassifier(random_state=seed)
                model4 = MLPClassifier(random_state=seed)
                 
                if y.unique().shape[0]>2:

                    model5 = XGBClassifier(seed=seed, objective="multi:softmax", num_class= y.unique().shape[0])
                else:
                    model5 =XGBClassifier(seed=seed)

                model6 = GaussianNB()
                
                for model, par, name_model in zip(
                    [model1, model2, model3, model4, model5, model6],

                    [
                        param_grids_lr,
                        param_grids_rf,
                        param_grids_dt,
                        param_grids_mlp,
                        param_grids_xgb,
                        param_grids_gs
                    ],
                    ["lr", "rf", "dt", "mlp", "xgb", "gs"]
                ):

                    if y.unique().shape[0] > 2:

                        model_search_cv = GridSearchCV(
                            model,
                            par,
                            cv=ps,
                            scoring="f1_macro",
                            n_jobs=-1,
                            refit=False,
                        )

                    else:
                        model_search_cv = GridSearchCV(
                            model, par, cv=ps, scoring="f1", n_jobs=-1, refit=False
                        )

                    model_search_cv.fit(X, y)

                    temp_r = pd.DataFrame(model_search_cv.cv_results_)
                    temp_r["Model"] = name_model
                    temp = temp.append(temp_r)
               
                scaler = MinMaxScaler()
                scaler.fit(train[num_cols])
                original = scaler.transform(train[num_cols])

                if title == "train":
                    dataset = test

                searcher = scaler.transform(dataset[num_cols])

                
                nbrs = NearestNeighbors(n_neighbors=10, n_jobs =-1).fit(original)

                distances, indices = nbrs.kneighbors(searcher)

                del original, searcher


                if num_flag:

                    metrics = evaluate(
                        dataset, train, metrics=["KSTest"], aggregate=False
                    )["raw_score"]


                else:
                    metrics = evaluate(
                        dataset, train, metrics=["KSTest", "CSTest"], aggregate=False
                    )["raw_score"]

                    temp["CSTest"] = metrics[1]
                
                temp["KSTest"] = metrics[0]
                temp["Mean_Knn"] = np.nanmean(distances[:, 0].mean())
                temp["10_Knn"] = np.nanquantile(
                    [distances[:, 0]], 0.1
                )

                temp["Mean_Ratio_Knn"] = np.nanmean((distances[:, 0] / distances[:, 1]))
                temp["Mean_Ratio_10Knn"] = np.nanmean((distances[:, 0] / distances[:, 9]))

                temp["10_Ratio_Knn"] = np.nanquantile(
                    (distances[:, 0] / distances[:, 1]), 0.1
                )          
                
                temp["10_Ratio_10Knn"] = np.nanquantile(
                    (distances[:, 0] / distances[:, 9]), 0.1
                )
                
                count_dup = pd.DataFrame(distances[:, 0])

                temp["Duplicates"] = count_dup[count_dup[0] == 0].shape[0]            
                

                temp["Generator"] = title
                temp["Seed"] = seed
                temp["fold"] = counter

                temp["phik_corr"] = (
                    (
                        dataset.phik_matrix(interval_cols=num_cols)
                        - train.phik_matrix(interval_cols=num_cols)
                    )
                    .abs()
                    .mean()
                    .mean()
                )

                storage = storage.append(temp)
                storage.reset_index(drop=True, inplace=True)
                
            time2 = time.time()

            pd.Series(time2 - time1).to_csv(str(counter) + ".csv")

            counter += 1

    storage.to_csv(iter_dataset + ".csv")

