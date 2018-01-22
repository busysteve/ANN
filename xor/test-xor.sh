#../ann -w xor-test-weights.xml -t xor_train.txt -x 100 -e 1000 -r 0.05 -m 0.02 -l T2 S3 T1 ; ../ann -w xor-test-weights.xml -i xor_input.txt
#../ann -E -w xor-test-weights.xml -t xor_train.txt -x 1 -e 200000 -r 0.008 -m 0.06 -l S2 S5 S1; ../ann -w xor-test-weights.xml -i xor_input.txt
#../ann -w xor-test-weights.xml -t xor_train.txt -x 1 -e 200000 -r 0.008 -m 0.06 -l S2 S5 S1; ../ann -w xor-test-weights.xml -i xor_input.txt
../ann -w xor-test-weights.xml -t xor_train.txt -x 1 -e 10000 -r 0.8 -m 0.6 -l S2 S5 S1; ../ann -w xor-test-weights.xml -i xor_input.txt
