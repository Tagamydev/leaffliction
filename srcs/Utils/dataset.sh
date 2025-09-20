#!/bin/bash

find ./images -type f | while read -r file; do
    type=$(echo "$file" | tr '/' ' ' | awk '{print $3}')
    echo "$file, $type"
done

