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
