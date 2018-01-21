./dataview 0 60000 10 255 "1 " "0 " train-images.idx3-ubyte train-labels-idx1-ubyte | awk -f prep-output.awk > training-data.txt

