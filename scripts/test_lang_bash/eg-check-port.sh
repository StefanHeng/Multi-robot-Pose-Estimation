#!/bin/bash
host="www.cyberciti.biz"
# echo " 443
# www.cyberciti.biz 442
# www.cyberciti.biz 444" | \
# while read  port; do
for port in {442..444}; do
#   r=$(bash -c 'exec 3<> /dev/tcp/'$host'/'$port';echo $?' 2>/dev/null)
#   r=$(bash -c 'exec 3<> nc -vz '$host' '$port';echo $?' 2>/dev/null)
#   echo "$r"
#   if [ "$r" = "0" ]; then
#     echo $host $port is open
#   else
#     echo $host $port is closed
#   fi
    # touch aa 2> /dev/null

    echo $host $port
    # nc -vz '$host' '$port' 2>/dev/null
    cmd="nc -vz ${host} ${port}"
    echo $cmd
    $cmd 2>/dev/null

    # if [ $? -eq 0 ]; then
    #     echo "${host} ${port} is open"
    # else
    #     echo "${host} ${port} is closed" >&2
    # fi
done
