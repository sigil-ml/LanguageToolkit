#!/bin/bash

echo -e "\n"

curl -i \
    -H "Content-type: application/json" \
    -H "Accept: application/json" \
    http://0.0.0.0:7000/health

echo -e "\n"
