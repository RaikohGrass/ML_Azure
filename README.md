# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
From the name of the data we can assume that data comes from a bank's adversiment plan. Whether it comes from the results or the people doing it cannot be stated.
I would assume it comes from the target parcipants as it includes personal data about them like age, marital status, education level and so on.
The objective of the project is to create a pipeline to create models predicting "Y" (I assume it marks whether the campaign was succesfull or not)

We implemented two pipelines, one using Hyperdrive and logistic regression and one using autoML.
The best result was obtained using autoML. It resulted in a VotingEnsemble with an accuracy of 0.91648
The best run of the hyperdrive gave us an accuracy of 0.91128, which is not far away from the best model

## Scikit-learn Pipeline
For the pipeline of Hyperdrive we defined a curated environment which included all the libraries needed for the training of a logistic regression model.
The training itself was performed with a predefined script that included the gathering of the data from a URL, its cleaning, and the divsision into training and test sets of it. 

After defining the environment a ScriptRunConfig was stablished. It encapsulated the environment, the target training script and the compute to be used

Having a defined ScriptRunConfig we can define the HyperDriveConfig object. The idea of HyperDrive is to create multiple instances of the same model (in our case a Logistic Regression) which are trained using different parameters each time (these are the hyperparameters, which are set fix before the training). HyperDrive will take our define search space (the sampler used) and iterate through it training a model each time and measuring its accuracy. 
This automated optimization process is really helpful as the tuning per hand of the hyperparameters is normally a time consuming task. 

The sampler used in this project allows us to define a fixed set of parameters which I want to iterate through. That way I know beforehand which values I'll expect and the search space can be restricted to a range based on previous knowledge, i.e., the algorithm won´t waste resources training models with parameter values out of a plausibel range.

The policy allows us to stop the hyperdrive run when no changes are noticed in the accuracy. In my particular case it didnt matter as all the combinatios were tried out due to the fixed range of the parameters.

## AutoML
Using autoML we defined a pipeline that gave us a total of 10 models. 
It didnt generate more as it didnt see a performance increase after training the VotinEnsemble
The VotingEnsemble is formed by seven algorithms. Each of them has an assigned weight which is using to regulate their saying for the prediction. 

For the autoML pipeline the following configuration was used:

```
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task='classification',
    primary_metric='accuracy',
    training_data=ds,
    label_column_name='y',
    n_cross_validations=2)
```
In the configuration the following parameters are defined:
- experiment_timeout_minutes: how long will azure wait during the response of the experiment before it breaks the request
- task: which kind of task we are working on (classification or regression). This will decide which kind of models will be trained
- primary_metric: the measurement that autoML should use to benchmark the models
- training_data: data to be used
- label_column_name: the objective of our regression or classification problem (included in "training_data")
- n_cross_validations: how many cross validations should be performed after training each model


## Pipeline comparison
There was just a small difference in accuracy and comparing the time required for the hyperdrive to compute, it was not worth it in this particular case.
The Hyperdrive pipeline was computed much faster. 


## Future work
It would be nice to know if increasing the range of the search space for the hyperdrive would bring any increase on the performance of the system. Since we are using a restricted search space it can be possible that our hyperdrive run get stuck in a local optimum. Making the search space wider could brin us closer to the global optimum for our problem. 
I would also like to know if autoML can make an optimized LogisticRegression and compare it to the one obtained with Hyperdrive

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
![image](https://user-images.githubusercontent.com/83981857/149114003-a3f69fb3-7bc1-4e25-afb6-5d4338a8345f.png)
