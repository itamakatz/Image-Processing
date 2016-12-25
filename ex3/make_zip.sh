#!/bin/bash
cd /home/itamar/Documents/ip/ex3/files/presubmition/
rm -f ex3.zip

cd /home/itamar/Documents/ip/ex3/submition/
zip -r /home/itamar/Documents/ip/ex3/files/presubmition/ex3.zip ./*

cd /home/itamar/Documents/ip/ex3/files/presubmition/
python3 ./test ./ex3