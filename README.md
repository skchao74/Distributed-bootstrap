# Distributed-bootstrap
    PYSPARK_PYTHON=python3 ARROW_PRE_0_15_IPC_FORMAT=1 spark-submit \
                  --master yarn  \
                  
Yarn: HDFS distributes file system 
Local: no distributed system

The gap with the current paper setting:
Synchronous parallelism: if the number of cores is insufficient (Spark), the parallelism becomes asynchronous
Communication cost in Spark scale with the number of nodes, not the number of partitions. 

The communication cost:
paper: tau_min * f(K)
spark: tau_min * f(node)
  Accessing partition - Yarn: intra-node; local: inter-node (more costly)
  Communication - Yarn: 
