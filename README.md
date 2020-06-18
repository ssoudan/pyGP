# pyGP

Various experimentations around Gaussian Processes.

## Run

    conda create --name pyGP python=3.7
    conda activate pyGP
    pip install -r requirements.txt
    
and then 'run' the main's in example/ from PyCharm.

or wtih docker:
    
    ./BUILD.sh && ./RUN.sh
    
You'll find the output in ... `output` folder.
        
## What is in there?

Don't expect too much originality here, it's mostly a bunch of tutorial or example code put together in the same place.

[GP regression](https://en.wikipedia.org/wiki/Kriging):
- src/main/python/run_gpflow.py: MLE and MCMC using GPflow - https://gpflow.readthedocs.io/en/stable/notebooks/regression.html
- src/main/python/run_sklearn.py: MLE using Scikit-learn - https://scikit-learn.org/stable/ 
- src/main/python/run_tfp.py: MLE and HMC using Tensorflow probability - https://www.tensorflow.org/probability

[Bayesian optimization](https://en.wikipedia.org/wiki/Bayesian_optimization):
- src/main/python/run_bo.py: comparison of the different variantes for BO purpose 

