def oversample_cross_validation(model, data):
    """Print the average performance measures after cross-validation (5 folds).
       Oversampling during cross-validation
       
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

        #-Oversamping before cross-validation -------------------------------------------

        # Use SMOTE to oversample training set.
        sm = SMOTE(random_state = 42, ratio = 1)
        over_X_train, over_y_train = sm.fit_sample(X_train.iloc[train_index], 
                                                   y_train.iloc[train_index])

        
        #-Fit model and prediction-------------------------------------------------------

        model.fit(pd.DataFrame(over_X_train, columns = X_train.columns), over_y_train)
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