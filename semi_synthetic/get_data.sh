#! /bin/bash

# Download and decompress data files

for year in {1987..2008..1}
do
    wget http://stat-computing.org/dataexpo/2009/${year}.csv.bz2
    bzip2 -d ${year}.csv.bz2
done

# Merge data files

for i in {1987..2008..1}
do
    awk -F, 'NR >= 2' ${i}.csv >> allfile_ordered_no_head.csv
    echo $i is processed
done

sed -i '1 i\Year,Month,DayofMonth,DayOfWeek,DepTime,CRSDepTime,ArrTime,CRSArrTime,UniqueCarrier,FlightNum,TailNum,ActualElapsedTime,CRSElapsedTime,AirTime,ArrDelay,DepDelay,Origin,Dest,Distance,TaxiIn,TaxiOut,Cancelled,CancellationCode,Diverted,CarrierDelay,WeatherDelay,NASDelay,SecurityDelay,LateAircraftDelay' allfile_ordered.csv

# Set up PySpark

module load anaconda
module load use.own
module load conda-env/mypackages-py3.7.0
module load anaconda/5.3.1-py37
module load spark/2.4.4
export ARROW_PRE_0_15_IPC_FORMAT=1
pyspark --master local[*]
