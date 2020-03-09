import argparse

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

import lightgbm as lgb
import matplotlib as mpl

import mlflow
import mlflow.lightgbm

mpl.use('Agg')


def parse_args():
    parser = argparse.ArgumentParser(description='LightGBM example')
    parser.add_argument('--max_no_depth', type=int, default=1.0,
                        help='maximum depth of the tree')
    parser.add_argument('--num_of_leaves', type=int, default=1.0,
                        help='number of leaves for tree model')
   
    parser.add_argument('--get_learning_rate', type= float, default=0.001, help= 'it is the learning rate at each step of your model optimization')
    return parser.parse_args()


def main():
    # parse command-line arguments
    args = parse_args()
    data = pd.read_csv('flights.csv')
    data = data.sample(frac = 0.1, random_state=10)

    data = data[["MONTH","DAY","DAY_OF_WEEK","AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT",
                     "ORIGIN_AIRPORT","AIR_TIME", "DEPARTURE_TIME","DISTANCE","ARRIVAL_DELAY"]]
    data.dropna(inplace=True)

    data["ARRIVAL_DELAY"] = (data["ARRIVAL_DELAY"]>10)*1

    cols = ["AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]
    for item in cols:
        data[item] = data[item].astype("category").cat.codes +1


    train, test, y_train, y_test = train_test_split(data.drop(["ARRIVAL_DELAY"], axis=1), data["ARRIVAL_DELAY"],
                                                    random_state=10, test_size=0.25)

    # enable auto logging
    mlflow.lightgbm.autolog()

    with mlflow.start_run():
        lg = lgb.LGBMClassifier(silent=False)
   

        d_train = lgb.Dataset(train, label=y_train)
        params = {"max_depth": args.max_no_depth, "learning_rate" : args.get_learning_rate, "num_leaves": args.num_of_leaves,  "n_estimators": 300}

        #With Catgeorical Features
        cate_features_name = ["MONTH","DAY","DAY_OF_WEEK","AIRLINE","DESTINATION_AIRPORT",
                         "ORIGIN_AIRPORT"]
        model2 = lgb.train(params, d_train, categorical_feature = cate_features_name)
        y_pred = model2.predict(test)
        loss = log_loss(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred.round())



        # log metrics
        mlflow.log_metrics({'log_loss': loss, 'accuracy': acc})


if __name__ == '__main__':
    main()

