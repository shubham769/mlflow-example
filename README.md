# LightGBM Example

This example trains a LightGBM classifier with the iris dataset and logs hyperparameters, metrics, and trained model.

## Running the code

```
python train.py --max_no_depth 50 --get_learning_rate 0.1 --num_of_leaves 900 ```

## Running the code as a project

```
mlflow run . -P max_no_depth 50  -P get_learning_rate 0.4 -P num_of_leaves 900
```







