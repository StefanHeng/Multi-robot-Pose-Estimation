#!/bin/bash
Timeout=1
host="www.cyberciti.biz"

for port in {442..444}; do
    # echo $host $port
    cmd="nc -vz ${host} ${port}"
    # echo $cmd
    ( $cmd ) & pid=$!
    ( sleep $Timeout
    gt&& kill -HUP $pid ) 2>/dev/null & watcher=$!
    if wait $pid 2>/dev/null; then
        case $? in
        0)
            echo "Exit with $? =======> ${host} ${port} is open";;
        1)
            echo "Exit with $? =======> ${host} ${port} is closed";;
        *)
            echo "Exit with $?";;
        esac
        pkill -HUP -P $watcher
        wait $watcher
    else
        echo "Exit with $? =======> ${host} ${port} timeout"
    fi
  done
