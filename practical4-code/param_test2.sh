#!/bin/bash

for i in `seq 1 5`;
        do
		echo $i
		echo "Gamma 0.1"
		python /home/isaacxia/Dropbox/cs181/Practicals/Practical\ 4/Practical4/practical4-code/param_test.py $i 0.1
		echo "Gamma 0.3"
		python /home/isaacxia/Dropbox/cs181/Practicals/Practical\ 4/Practical4/practical4-code/param_test.py $i 0.3
		echo "Gamma 0.4"
		python /home/isaacxia/Dropbox/cs181/Practicals/Practical\ 4/Practical4/practical4-code/param_test.py $i 0.4
        done 

