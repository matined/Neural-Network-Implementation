#Simple Neural Network Implementation
I do this project to better understand how things work. I'm going to add new features and optimize it as I learn new things and concepts.

## Libraries
+ numpy

## How to use?
To use the model you have to create class NN_model object, then, you can use `fit` and `predict()` methods.
```python
model = NN_model(layers_sizes=(3, 5, 3, 1), learning_rate=0.075, max_iter=200, activation='relu', verbose=True)
model.fit(X_train, y_train)
model.predict(X_test)
```
