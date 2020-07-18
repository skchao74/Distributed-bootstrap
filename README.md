

### Idea of spark:  

    PYSPARK_PYTHON=python3 ARROW_PRE_0_15_IPC_FORMAT=1 spark-submit \
                  --master yarn  \
                  
yarn: HDFS distributes file system  
local: no distributed system

0) .sh file to request nodes in Slurp
1) Initiating spark. yarn: use HDFS to store data on nodes; local: save data on only one node;  
2) Spark.py: repartition data to desirable partition size. The partitions will be processed with the cores in the node, and one writes the operations that each core does to each partition.

### The gap between Spark and the master-worker setting in the paper:  
Synchronous parallelism: if the number of cores is insufficient (Spark), the parallelism becomes asynchronous  
Communication cost in Spark scale with the number of nodes, not the number of partitions

### The communication cost:  
paper: tau_min * f(k). k is the number of machines, which is analogous to the number of partitions in Spark.  
spark: tau_min * f(#node). Each node may save multiple partitions. So the actual communication cost may be better than the setting considered in the paper because #node < k (#partition)

### The processing cost can be decomposed to two parts:
Accessing partition - yarn: intra-node; local: inter-node (more costly)  
Communication - yarn: inter-node(?); local: ??? we are not sure whether the result is saved in the central place where the data are saved, or on the node where the core is

### References
https://github.com/feng-li/dlsa  
