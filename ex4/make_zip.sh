#!/bin/bash
cd /home/itamar/Documents/ip/ex4/files/presubmition/
rm -f ex4.zip

cd /home/itamar/Documents/ip/ex4/submition/
zip -r /home/itamar/Documents/ip/ex4/files/presubmition/ex4.zip ./*

cd /home/itamar/Documents/ip/ex4/files/presubmition/
python3 ./test ./ex4