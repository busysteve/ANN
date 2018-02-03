g++ -std=c++11 -g -o tann tANN.cpp ../XMLTag/xmltag.cpp -I.. -I../thrust -pthread -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
