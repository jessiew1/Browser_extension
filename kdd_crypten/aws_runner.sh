# The variables at the top of this file should be configured before running the file.
# Make sure to read ObtainingCredentialsFile.txt before running this script.

# These are the Instance IDs of the three AWS servers.
instance1=i-0ec43f375fc41d5f0
instance2=i-00facb28e36510d1b
instance3=i-0a07806be55f0be04

# These are the private key files for each of the three AWS servers.
key1=kp21.pem
key2=kp21.pem
key3=kp21.pem

# These are the regions of the three AWS servers.
# Look at the server's availability zone to find out the region.
# If putting the availability zone here does not work, try writing something less specific.
# For example, availability zone us-east-1d becomes region us-east-1.
region1=us-east-1
region2=us-east-1
region3=us-east-1

# This is the credentials file obtained from ObtainingCredentialsFile.txt.
credentials=./credentials

# The python program that contains the crypten computation to run.
script_name=KDDSortingNetwork.py

# Additional files that should be uploaded to the AWS servers so that the python program can run properly.
aux_files=sorting_network.py,compute_server_io.py,Workspace.py

python aws_launcher.py --regions=$region1,$region2,$region3 --ssh_key_file=$key1,$key2,$key3 --instances=$instance1,$instance2,$instance3 --credentials=$credentials --aux_files=$aux_files $script_name
