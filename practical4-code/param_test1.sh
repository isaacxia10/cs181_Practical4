#!/bin/bash

for i in `seq 1 5`;
        do
		echo $i
		echo "Gamma 0.5"
		python /home/isaacxia/Dropbox/cs181/Practicals/Practical\ 4/Practical4/practical4-code/param_test.py $i 0.5
		echo "Gamma 0.7"
		python /home/isaacxia/Dropbox/cs181/Practicals/Practical\ 4/Practical4/practical4-code/param_test.py $i 0.7
		echo "Gamma 1"
		python /home/isaacxia/Dropbox/cs181/Practicals/Practical\ 4/Practical4/practical4-code/param_test.py $i 1 
        done 

