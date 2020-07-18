# Distributed-bootstrap
    PYSPARK_PYTHON=python3 ARROW_PRE_0_15_IPC_FORMAT=1 spark-submit \
                  --master yarn  \
                  
yarn: HDFS distributes file system  
local: no distributed system

The gap between Spark and the master-worker setting in the paper:  
Synchronous parallelism: if the number of cores is insufficient (Spark), the parallelism becomes asynchronous  
Communication cost in Spark scale with the number of nodes, not the number of partitions

The communication cost:  
paper: tau_min * f(K)  
spark: tau_min * f(node)  

Accessing partition - yarn: intra-node; local: inter-node (more costly)  
Communication - yarn: inter-node(?); local: ??? we are not sure whether the result is saved in the central place where the data are saved, or on the node where the core is
