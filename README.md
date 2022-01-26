# Simple Neural Network Implementation
I do this project for the better understanding of NN. I'm going to add new features and optimize it as I learn new things and concepts.

## Libraries
+ numpy

## How to use?
To use the model you have to create a class `NN_model` object, then, you can use methods `fit()` and `predict()`.
```python
model = NN_model(layers_sizes=(3, 5, 3, 1), learning_rate=0.075, max_iter=200, activation='relu', verbose=True)
model.fit(X_train, y_train)
model.predict(X_test)
```
`X_train`  &nbsp; the training set, numpy array of shape (number_of_examples, number_of_features) <br>
`y_train`  &nbsp; target values for the training set, numpy array of shape (number_of_examples, target) <br>
`X_test`  &nbsp; &nbsp; the test set, numpy array of shape (number_of_examples, number_of_features)

`NN_model` class is in the NN_model.py file.

## Status
Features I've already added:
+ random initailization
+ forwad propagation
+ backward propagation
+ computing cross-entrop cost
+ activation functions
  + reLU
  + sigmoid 
