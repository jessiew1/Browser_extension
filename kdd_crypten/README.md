# kdd_crypten

An implementation of the KDD algorithms in an MPC environment using Crypten.

| File | Description |
| :--- | :--- |
| README.md | This file. |
| aws_launcher.py | Used to upload python programs to AWS servers and run the Crypten computations. See aws_runner.sh for instructions. |
| aws_runner.sh | A bash script for running a Crypten python program on three AWS servers. |
| Benchmark.py | Compares the cost of computing thresholds using sorting networks and merge sort. To be used in conjunction with BenchmarkInterpreter.py. |
| benchmark_results.1 | The output of Benchmark.py that is to be used as input to BenchmarkInterpreter.py |
| BenchmarkInterpreter.py | Displays the results of Benchmark.py in a graph. |
| BenchmarkInterpreterOutput.png | The picture that BenchmarkInterpreter.py displays. |
| compute_server_io.py | Utility functions for handling compute party input/output. |
| ComputeKthLeast.py | Work in progress oblivious shuffle in Crypten. |
| crypten_local_runner.sh | A bash script for running a Crypten python program locally in three separate terminals. |
| encoder_scale_experiments.py | A very old program that was used to experiment with Crypten's encoder scales to see if saving and loading cryptensors is supported by the Crypten interface, instead of using undocumented functions to save and load cryptensors. |
| KDDSortingNetwork.py | An implementation of the KDD algorithms that uses a sorting network when computing thresholds. |
| KDDSortingNetworkRevealed.py | Like KDDSortingNetwork.py, but reveals the comparison results to see if there is a potential to improve runtime. Not secure because of the revealed comparison results. |
| KDDMergeSort.py | Work in progress converting KDDSortingNetwork.py to use merge sort instead of a sorting network, which will increase performance. Currently not secure because the oblivious shuffle is not implemented. |
| saving_cryptensors_demo.py | A short program that demonstrates saving and loading cryptensors by accessing its share property. |
| sorting_network.py | A sorting network implementation. |
| test_sorting_network.py | Code that tests sorting_network.py on a randomly shuffled input. |
| Workspace.py | Main python configuration file. |
| logs/ | Logs of KDDSortingNetwork.py, KDDSortingNetworkRevealed.py, and KDDMergeSort.py on AWS servers. Contains runtime information. See logs/README.md for more information. |

For python files, look at the top of the file for comments about what the program does.
Most python files that are intended to be run do not need additional command line arguments.
