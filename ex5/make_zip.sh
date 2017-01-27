#!/bin/bash
cd /home/itamar/Documents/ip/ex5/files/presubmition/
rm -f ex5.zip

cd /home/itamar/Documents/ip/ex5/submition/
zip -r /home/itamar/Documents/ip/ex5/files/presubmition/ex5.zip ./*

cd /home/itamar/Documents/ip/ex5/files/presubmition/
python3 ./test ./ex5