#!/bin/bash
cd /home/itamar/Documents/ip/ex2/files/presubmition/
rm -f ex2.zip
rm -f /home/itamar/Documents/ip/ex2/submition/sol2
cp /home/itamar/Documents/ip/ex2/sol2.py /home/itamar/Documents/ip/ex2/submition/

cd /home/itamar/Documents/ip/ex2/submition/
zip -r /home/itamar/Documents/ip/ex2/files/presubmition/ex2.zip ./*


cd /home/itamar/Documents/ip/ex2/files/presubmition/
python3 ./test ./ex2