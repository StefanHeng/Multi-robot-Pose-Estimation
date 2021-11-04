#!/bin/bash
echo "scanme.nmap.org 80
scanme.nmap.org 81
192.168.0.100 1" | (
  TCP_TIMEOUT=3
  while read host port; do
    echo $host $port
    (CURPID=$BASHPID;
    (sleep $TCP_TIMEOUT;kill $CURPID) &
    exec 3<> /dev/tcp/$host/$port
    ) 2>/dev/null
    case $? in
    0)
      echo $host $port is open;;
    1)
      echo $host $port is closed;;
    143) # killed by SIGTERM
       echo $host $port timeouted;;
    *)
        echo $?;;
    esac

  done
  ) 2>/dev/null # avoid bash message "Terminated ..."
