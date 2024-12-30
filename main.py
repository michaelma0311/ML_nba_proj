import pandas as pd
import warnings
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#functions
def check(data, model, predictors, start=2, step=1):
    all_predic = []
    seasons = sorted(data["season"].unique())
    for i in range(start, len(seasons), step):
        cursea = seasons[i]
        train = data[data["season"] < cursea]
        test=data[data["season"] == cursea]
        model.fit(train[predictors], train["target"])
        preds = pd.Series(model.predict(test[predictors]), index=test.index)
        comb = pd.concat([test["target"], preds], axis=1)
        comb.columns = ["actual", "prediction"]
        all_predic.append(comb)
    return pd.concat(all_predic)
def shifts(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(df, col_name):
    return df.groupby("team", group_keys=False).apply(lambda x: shifts(x, col_name), include_groups=False)

def checkavg(cur):
    rolling = cur.rolling(15).mean()
    return rolling

def adnew(group):
    group["target"] = group["won"].shift(-1)
    return group

#data cleaning

df = pd.read_csv("nbadata.csv", index_col=0)
df = df.sort_values("date")
df = df.reset_index(drop=True)
#predicting the winner of this new game
new_row = {
    "mp": 48, "mp.1": 48, "fg": 18, "fga": 40, "fg%": 0.45, "3p": 10, "3pa": 25, "3p%": 0.4, 
    "ft": 15, "fta": 20, "ft%": 0.75, "orb": 10, "drb": 25, "trb": 35, "ast": 25, "stl": 10, 
    "blk": 5, "tov": 15, "pf": 20, "pts": 61, "+/-": 5, "ts%": 0.57, "efg%": 0.5, "3par": 0.25,
    "ftr": 0.2, "orb%": 0.15, "drb%": 0.25, "trb%": 0.2, "ast%": 0.3, "stl%": 0.1, "blk%": 0.05, 
    "tov%": 0.12, "usg%": 0.22, "ortg": 110, "drtg": 105, "mp_max": 42, "mp_max.1": 42, "fg_max": 8, 
    "fga_max": 18, "fg%_max": 0.45, "3p_max": 5, "3pa_max": 13, "3p%_max": 0.38, "ft_max": 7, "fta_max": 9, 
    "ft%_max": 0.78, "orb_max": 3, "drb_max": 8, "trb_max": 11, "ast_max": 8, "stl_max": 3, "blk_max": 2, 
    "tov_max": 5, "pf_max": 6, "pts_max": 23, "+/-_max": 7, "ts%_max": 0.58, "efg%_max": 0.5, "3par_max": 0.24, 
    "ftr_max": 0.21, "orb%_max": 0.18, "drb%_max": 0.27, "trb%_max": 0.22, "ast%_max": 0.31, "stl%_max": 0.11, 
    "blk%_max": 0.06, "tov%_max": 0.13, "usg%_max": 0.24, "ortg_max": 113, "drtg_max": 102, "team": "GSW", 
    "total": 100, "home": 1, "index_opp": 0, "mp_opp": 48, "mp_opp.1": 48, "fg_opp": 17, "fga_opp": 40, 
    "fg%_opp": 0.425, "3p_opp": 9, "3pa_opp": 23, "3p%_opp": 0.39, "ft_opp": 14, "fta_opp": 20, "ft%_opp": 0.7, 
    "orb_opp": 9, "drb_opp": 23, "trb_opp": 32, "ast_opp": 22, "stl_opp": 8, "blk_opp": 4, "tov_opp": 12, 
    "pf_opp": 19, "pts_opp": 57, "+/-_opp": -5, "ts%_opp": 0.53, "efg%_opp": 0.47, "3par_opp": 0.23, 
    "ftr_opp": 0.19, "orb%_opp": 0.14, "drb%_opp": 0.24, "trb%_opp": 0.19, "ast%_opp": 0.28, "stl%_opp": 0.09, 
    "blk%_opp": 0.04, "tov%_opp": 0.11, "usg%_opp": 0.21, "ortg_opp": 108, "drtg_opp": 110, "mp_max_opp": 41, 
    "mp_max_opp.1": 41, "fg_max_opp": 7, "fga_max_opp": 17, "fg%_max_opp": 0.41, "3p_max_opp": 4, "3pa_max_opp": 10, 
    "3p%_max_opp": 0.4, "ft_max_opp": 6, "fta_max_opp": 8, "ft%_max_opp": 0.75, "orb_max_opp": 2, "drb_max_opp": 7, 
    "trb_max_opp": 9, "ast_max_opp": 7, "stl_max_opp": 2, "blk_max_opp": 1, "tov_max_opp": 4, "pf_max_opp": 5, 
    "pts_max_opp": 18, "+/-_max_opp": -4, "ts%_max_opp": 0.54, "efg%_max_opp": 0.46, "3par_max_opp": 0.22, 
    "ftr_max_opp": 0.18, "orb%_max_opp": 0.13, "drb%_max_opp": 0.22, "trb%_max_opp": 0.18, "ast%_max_opp": 0.27, 
    "stl%_max_opp": 0.08, "blk%_max_opp": 0.05, "tov%_max_opp": 0.1, "usg%_max_opp": 0.2, "ortg_max_opp": 112, 
    "drtg_max_opp": 109, "team_opp": "WAS", "total_opp": 97, "home_opp": 0, "season": 2022, "date": "2022-12-31", "won":1, "target": 1
}

new_row_df = pd.DataFrame([new_row])

df = pd.concat([df, new_row_df], ignore_index=True)
df["team1"] = df["team"]
df = df.groupby("team1", group_keys = False).apply(adnew, include_groups=False).reset_index(drop=False)


df.loc[pd.isnull(df["target"]), "target"] = 2

df["target"] = df["target"].astype(int, errors="ignore")
curnull = pd.isnull(df)
curnull = curnull.sum()
curnull = curnull[curnull > 0]
valid_cols = df.columns[~(df.columns.isin(curnull.index))]
df = df[valid_cols].copy()

#actual machine learning

rr = RidgeClassifier(alpha=1)

split = TimeSeriesSplit(n_splits=3)

sfs = SequentialFeatureSelector(rr, n_features_to_select=14, direction="forward",cv=split)

rev_cols = ["season", "date", "won", "target", "team", "team_opp"]

sel_cols = df.columns[~df.columns.isin(rev_cols)]


scaler = MinMaxScaler()
df[sel_cols] = scaler.fit_transform(df[sel_cols])
sfs.fit(df[sel_cols], df["target"])

predictors = list(sel_cols[sfs.get_support()])


predictions = check(df, rr, predictors)

df["home1"] = df["home"]
df.groupby("home1").apply(lambda x: x[x["won"]==1].shape[0] / x.shape[0], include_groups=False)

rollingvals = df[list(sel_cols) + ["won", "team", "season"]]


rollingvals = rollingvals.groupby(["team", "season"], group_keys=False).apply(checkavg, include_groups = False)

rolling_cols = ["{}_10".format(i) for i in rollingvals.columns]
rollingvals.columns = rolling_cols

df = pd.concat([df, rollingvals], axis=1)

df = df.dropna()


df.loc[:, 'home_next'] = add_col(df, 'home')
df.loc[:, 'team_opp_next'] = add_col(df, 'team_opp')
df.loc[:, 'date_next'] = add_col(df, 'date')

temp = df.merge(df[rolling_cols + ["team_opp_next", "date_next", "team"]], left_on = ["team", "date_next"], right_on=["team_opp_next", "date_next"])

rev_cols += list(temp.columns[temp.dtypes == "object"])

sel_cols = temp.columns[~(temp.columns.isin(rev_cols))]

sfs.fit(temp[sel_cols], temp["target"])

predictors = list(sel_cols[sfs.get_support()])


predictions = check(temp, rr, predictors)
last_prediction = predictions["prediction"].iloc[-1]
print("accruacy score", accuracy_score(predictions["actual"], predictions["prediction"]))
if (last_prediction == 1):
    print("{} is predicted to win against {}.".format(df.iloc[-1]["team"], df.iloc[-1]["team_opp"]))
else:
    print("{} is predicted to win against {}.".format(df.iloc[-1]["team_opp"], df.iloc[-1]["team"]))

