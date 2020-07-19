# Oversampling-Undersampling-CrossValidation

There are a lot of unbalanced data in the real world. Dealing with extremely unbalanced data needs some extra efforts. This March I completed a credit card fraud detection data challenge from Capital One, which was the 3rd stage of the whole interview. I passed the data challenge, but unfortunately due to the pandemic, the position rescinded. However, I learned a lot from this data challenge. 

For fraud detection problem, it is usually unbalanced because most of the transactions are normal in the real world. But it is important to detect a fraudulent transaction for banks to protect customers. There are couple of ways to deal with unbalanced data, oversampling and undersampling are some popular ones. The functions are mainly about oversampling/undersampling during cross validations. The concept is: always sampling on the training set after train test split, so does cross validation.

At the end of the function, I print out different metrics, such as accuracy, recall and precision......etc. You can decide which metrics you care the most, and optimize your models and hyperparameters to maximize these metrics.
