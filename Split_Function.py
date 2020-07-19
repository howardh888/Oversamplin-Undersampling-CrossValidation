import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data):
    """ Split the dataset to predictors and label, 
        then split them into training and testing set.
        
        data: fill with a dataset"""
    
    # Assign predictors to X, target variable to y.
    X = data.drop('isFraud', axis=1)
    y = data['isFraud']

    # Split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.3, 
                                                        random_state = 42)
    return (X_train, X_test, y_train, y_test)