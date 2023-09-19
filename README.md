# CryptenProject

## Repository structure

| File | Description |
| :--- | :--- |
| README.md | This file. |
| AWSRoutingServer/ | Code for the routing server, made using Express. |
| AWSComputeServer/ | Additional files pertaining to running the kdd_crypten code on AWS servers. |
| kdd_clear_reimplementation/ | Reimplementation of the KDD algorithms using more barebones math. |
| kdd_crypten/ | Implementation of the KDD algorithms in an MPC environment using Crypten. |
| kdd-code | KDD paper authors' code, possibly with minor modifications during our experimentation. |

For python files, look at the top of the file for comments about what the program does.
Most python files that are intended to be run do not need additional command line arguments.

## Instructions to run

### kdd_crypten locally

Crypten is only supported on Linux, so we run our code in a Ubuntu virtual machine using VirtualBox.

Move all the files in this repository to the virtual machine.

The remaining instructions are located in the comments at the top of kdd_crypten/crypten_local_runner.sh.

### kdd_crypten on AWS servers

The instructions are located in AWSComputeServer/ComputeServerInstallInstructions.txt.

### Routing server on AWS servers

The instructions are located in AWSRoutingServer/ExpressHTTPServer/ExpressAppInstructions.txt.

## Instructions to develop

### Routing server on AWS servers

The routing server was implemented using Express and MongoDB.

To modify the code of the routing server, you would likely want to modify the files AWSRoutingServer/ExpressHTTPServer/RoutingServerExpress/app.js and AWSRoutingServer/ExpressHTTPServer/RoutingServerExpress/routes/index.js.

Marek ran into issues with CORS. The app.js file now addresses that. The index.js file implements endpoints to test encrypting a share to each of the three compute parties.

Alternatively, you could choose to create another backend server from scratch. The routing server is currently mostly endpoints that test that encryption and decryption are compatible with Marek's extension. There are some functions to get random numbers, write to file, and write to the database.

## Miscellaneous notes

### Shell script line endings

When uploading .sh files to Linux machines, if when running the scripts it appears not to work properly and as though some commands that should have been run were not run, check that the line endings of the .sh file are all \n. When moving files from machine to machine or to and from git, they might be Windows line endings, which are \r\n, which will cause the shell script not to be parsed properly by the terminal.