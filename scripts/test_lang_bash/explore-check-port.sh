#!/bin/bash
Timeout=1
host="www.cyberciti.biz"

# echo "scanme.nmap.org 80
# scanme.nmap.org 81
# 192.168.0.100 1" | (
#   while read host port; do
for port in {442..444}; do
    # echo $host $port
    # cmd="/dev/tcp/${host}/${port}"
    cmd="nc -vz ${host} ${port}"
    # echo $cmd

    # (CURPID=$BASHPID;
    # (sleep $TCP_TIMEOUT;kill $CURPID) &
    # exec 3<> $cmd
    # ) 2>/dev/null

    ( $cmd ) & pid=$!
    ( sleep $Timeout && kill -HUP $pid ) 2>/dev/null & watcher=$!
    if wait $pid 2>/dev/null; then
        case $? in
        0)
            echo "Exit with $? =======> ${host} ${port} is open";;
        1)
            echo "Exit with $? =======> ${host} ${port} is closed";;
        *)
            echo "Exit with $?";;
        esac
        # echo "your_command finished"
        pkill -HUP -P $watcher
        wait $watcher
    else
        echo "Exit with $? =======> ${host} ${port} timeout"
        # echo $? "exit code"
        # echo "your_command interrupted"
    fi
    # case $? in
    # 0)
    #     echo "Exit with $? =======> ${host} ${port} is open";;
    # 1)
    #     echo "Exit with $? =======> ${host} ${port} is closed";;
    # 143) # killed by SIGTERM
    #     echo "Exit with $? =======> ${host} ${port} timeouted";;
    # *)
    #     echo "Exit with $?";;
    # esac

  done
#   ) 2>/dev/null # avoid bash message "Terminated ..."
