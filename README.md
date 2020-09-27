# kfold_validation
This is a file that can be run on a linux server with PBS to execute 10-fold validation of a CNN using a tfrecord that you build with the functions in the web_pipeline repository. It needs the following packages: tensorflow, yaml, os.

# parallel_job
This is a PBS script that will enable you to run the folds of the 10-fold validation in parallel.
In the script you define the job name and the simultaneously the number of frozen convolutional layers at the bottom of the CNN after "#PBS -N ".
