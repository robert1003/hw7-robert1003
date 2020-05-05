#!/bin/bash

if [ $# -ne 2 ]; then
  echo -e "usage:\tbash hw7_test.sh [input_image_directory] [pred_file]"
  exit
fi

python3 hw7_test.py $1 $2
