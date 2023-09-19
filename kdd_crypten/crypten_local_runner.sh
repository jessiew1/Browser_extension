# Runs a crypten script locally.
# To use, suppose you want to run a script using n compute parties.
# Open n terminals.
# In each terminal, run the command . crypten_local_runner.sh n i file_name, where n is the number of compute parties, i is a unique number from 0 to n-1 inclusive, and file_name is the name of the python file that contains the crypten code to be executed.

# Example:
# In the first terminal, run . crypten_local_runner.sh 3 0 KDDSortingNetwork.py
# In the second terminal, run . crypten_local_runner.sh 3 1 KDDSortingNetwork.py
# In the third terminal, run . crypten_local_runner.sh 3 2 KDDSortingNetwork.py

WORLD_SIZE=$1
RANK=$2
SCRIPT_NAME=$3
export WORLD_SIZE
export RANK
echo WORLD_SIZE is $WORLD_SIZE
echo RANK is $RANK
echo SCRIPT_NAME is $SCRIPT_NAME
echo
python3 $SCRIPT_NAME
