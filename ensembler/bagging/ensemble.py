'''
Created on 09-Mar-2016

@author: pankajrawat
'''


import os

import pandas as pd


def get_scaled_data(num, scale=1.28, is_scale=True):
    return scale ** (num * 100) if is_scale else num


"""
Function perform weighted average of each model predictions. Model with higher accuracy rate will have more weight to 
predicted probability
    Parameters
    ----------
    score_dict : dict
        dict pair of CSV: score (logloss).
    column_id: string, optional (default='ID')
        unique identifier for each row
    outputFile: string, optional (default='ensemble_byweight_logloss.csv')
        output csv which contains the ensembled log_loss
    scale: float, optional (default, 1.28)
        define the influence of dominating models on overall weight scale
        
"""
def ensemble_byweight_logloss(score_dict, column_id='ID', outputFile='ensemble_byweight_logloss.csv', scale=1.28):
    """
    This function assumes that logloss value would be in range (0, 1)
    """
    df = None
    df_id = None
    is_scale = True
    total_weight = reduce(lambda x, y: x + y, map(lambda x: get_scaled_data(1 - x, scale=scale, is_scale=is_scale), score_dict.values()))

    for key, score in score_dict.items():
        score = get_scaled_data(1 - score, is_scale=is_scale)
        #print score * 100, score_scale
        weight = score / total_weight
        print "N:", score, "  => ", weight
        df_csv = pd.read_csv(key)

        #print newDf
        if not isinstance(df, pd.DataFrame):
            df_id = df_csv[[column_id]]
            df_csv.drop([column_id], axis=1, inplace=True)
            df_csv = df_csv * weight
            df = df_csv
            continue

        df_csv.drop([column_id], axis=1, inplace=True)
        df_csv = df_csv * weight
        df = df.add(df_csv, fill_value=0)
    df = pd.concat([df_id, df], axis=1)
    df.to_csv(outputFile, index=False)

if __name__ == "__main__":
    column_id = 'ID'

    """ contains file, score pairs, these will be average
    """
    score_dict = {
                    os.path.join('input1.csv'): 0.47317,
                    os.path.join('input2.csv'): 0.45900,
             }
    ensemble_byweight_logloss(score_dict)

#df = pd.read_csv(os.path.join('..', 'input', 'extra_trees.csv'))
#df['PredictedProb'] = df['PredictedProb'].apply(lambda x: .98 if x > 0.9 else x)
#df['PredictedProb'] = df['PredictedProb'].apply(lambda x: 0.2 if x < 0.1 else x)

#df.to_csv("corrected.csv", index=False)
