# ANN
Artificial Neural Network C++ template based class set

Build with:

    g++ -O3 -o ann ANN.cpp XMLTag/xmltag.cpp


Lets look at the following command line:

    ./ann -w letters.weights.xml -i letters_input.txt -r 0.05 -m 0.0001 -t letters_train.txt -o 5003 -e 30000  -b -l L49 S10 L5

    -w = XML file to save weights into will be rewritten after each training
    -i = optional input data file for running against saved or trained weights
    -r = learning rate for training
    -m = momentum value for training
    -t = training file with inputs and outputs
    -o = progress output modulus
    -e = training epochs
    -b = flag to utilize bias
    -l = layers 
        L49 = Linear layer of 49 input nodes
        S10 = Sigmoid hidden layer of 10 nodes
        L5  = Linear output layer of 5 nodes
    
More from within alphabet:

    cat test_letters.txt | awk -f prep_letters_for_input.awk | ../ann -w letters.weights.xml -i - | awk -f translate_letter_binary.awk

