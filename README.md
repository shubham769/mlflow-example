# LightGBM Example

This example trains a LightGBM classifier with the flight delay dataset and logs hyperparameters, metrics, and trained model.
you can access dataset from here: https://bit.ly/33MhdU9

## Running the code 

```python train.py --max_no_depth 50 --get_learning_rate 0.1 --num_of_leaves 900 ```

keeep doing experiments by chaning hypermeters of the model by changing maxy_no_depth, get_learning_rate, num_of_leaves.
for example:

```python train.py --max_no_depth 100 --get_learning_rate 0.7 --num_of_leaves 1000 ```
```python train.py --max_no_depth 150 --get_learning_rate 0.5 --num_of_leaves 800 ```

And to keep ytrack of all the experiments just type:
```mlflow ui```

It will open a beautiful dashboard consist of all informations about experiments.

You can compare diffrent model also by looking at logged metrics , like error rate and accuracy.

## Running the code as a project

```mlflow run . -P max_no_depth 50  -P get_learning_rate 0.4 -P num_of_leaves 900 ```







