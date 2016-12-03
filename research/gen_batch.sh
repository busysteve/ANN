awk 'BEGIN{ print topology: 2 4 1; for(i=0; i<1000; i++){ print in: 1.0 1.0; print out: 0.0;} }' > /tmp/trainingData.txt
