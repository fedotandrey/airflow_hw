import pandas as pd
import os
import json
import dill


path = os.environ.get('PROJECT_PATH', '.')


def read_pkl():
    pkl_filename = f'{path}/data/models/cars_pipe.pkl'
    with open(pkl_filename, 'rb') as file:
        model = dill.load(file)
    return model


def predict():
    model = read_pkl()
    pred = {}
    for i in os.listdir(f'{path}/data/test'):
        with open(f'{path}/data/test/{i}', 'rb') as f:
            data = json.load(f)
        df = pd.DataFrame([data])
        age_category = model.predict(df)
        pred[df.loc[0, 'id']] =  age_category
    df_2 = pd.DataFrame(list(pred.items()),
                        columns=['car_id', 'pred'])
    df_2.to_csv(f'{path}/data/predictions/preds_202207191811.csv')


if __name__ == '__main__':
    predict()
