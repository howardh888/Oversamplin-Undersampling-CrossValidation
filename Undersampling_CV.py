def undersample_cross_validation(model, data):
    """Print the average performance measures after cross-validation (5 folds).
       Undersampling during cross-validation
       
       model: fill with a classifier
       data: fill with a dataset"""
    
    # Create empty list to append the output value from the list.
    accuracy_lst = []
    precision_lst = []
    recall_lst = []
    f1_lst = []
    auc_lst = []
    
    # Split the data to training and testing set.
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Set up the folds for cross-validation.
    skf = StratifiedKFold(n_splits = 5, random_state = 42)
    model = model 

    for train_index, test_index in skf.split(X_train, y_train): 

        #-Undersamping before cross-validation -------------------------------------------

        # Concat training set and label. Only undersample training set.
        train = pd.concat([X_train.iloc[train_index], y_train.iloc[train_index]], axis=1)

        # Separate the training set by the 2 classes of target variable.
        not_fraud = train[train['isFraud'] == 0]
        fraud = train[train['isFraud'] == 1]

        # Random select samples from not_fraud that matches the number of frauds.
        not_fraud_undersample = resample(not_fraud,
                                        replace = False,
                                        n_samples = len(fraud), # to match # of not_fraud
                                        random_state = 42)

        # Combine 2 dataset.
        undersample = pd.concat([not_fraud_undersample, fraud], axis = 0)

        # Separate predictors and label for undersampled training set.
        under_X_train = undersample.drop('isFraud', axis=1)
        under_y_train = undersample['isFraud']

        #-Fit model and prediction-------------------------------------------------------

        model.fit(under_X_train, under_y_train)
        prediction = model.predict(X_train.iloc[test_index])

        accuracy_lst.append(accuracy_score(y_train.iloc[test_index], prediction))
        precision_lst.append(precision_score(y_train.iloc[test_index], prediction))
        recall_lst.append(recall_score(y_train.iloc[test_index], prediction))
        f1_lst.append(f1_score(y_train.iloc[test_index], prediction))
        auc_lst.append(roc_auc_score(y_train.iloc[test_index], prediction))
  
    print('Cross-Validation Performance:')
    print("accuracy: {}".format(np.mean(accuracy_lst)))
    print("precision: {}".format(np.mean(precision_lst)))
    print("recall: {}".format(np.mean(recall_lst)))
    print("f1: {}".format(np.mean(f1_lst)))
    print("auc: {}".format(np.mean(auc_lst)))