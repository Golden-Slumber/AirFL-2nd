# AirFL-2nd
Simulation codes for over-the-air federated learning via second-order optimization.
## Prepare ( for logistic regression )
> * go to [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) to download datasets (e.g. covtype, a9a, w8a, phishing).
> * python txt2npz.py.
> * Now we have corresponding '.npz' files.
## Experiments 
> go to the directory "/Experiments" to run different demos, the results will be saved to "/Outputs".
## Algorithms 
> * the files under directory "/Algorithms" serve as the main logic of simulation, where solvers stand for the server and executors represent the devices.
> * DC releated codes are saved to the directory "/Utils".
