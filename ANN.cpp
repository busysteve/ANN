
// g++ -g -o ann ANN.cpp

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>

const void* nullptr = NULL;

template<typename T>
struct Node;


template<typename T>
struct Connection
{

    T weight, alpha, delta;

    Node<T> *toNode;

    Connection( Node<T>* node ) : toNode( node ), alpha((T)1.0), delta((T)0.0)
    {
        T rnd = (T)std::rand() / RAND_MAX;

        weight = ( rnd * (T)1.5 ) - ( (T)1.5 / (T)2.0 );
    }

	void xmit( T in )
	{
		if( toNode != nullptr )
        {
			toNode->input( in*weight ); // Apply weight here
            printf( " <%0.3f|%0.3f>(%f)\n", in, weight, in*weight );
        }
	}


};


template<typename T>
struct Node
{

    T inSum, lastOut;
    T deltaErr;
    T grad;

	std::vector<Connection<T>*> conns;
	std::vector<Connection<T>*> inConns;

    Node() : inSum((T)0.0), lastOut((T)0.0), deltaErr((T)0.0), grad((T)1.0) {}

	void input( T in )
    {
        inSum += in; // Sum weighted inputs for activation
    }

    T out()
    {
        return lastOut;
    }

    // Node to bind to (next layer node)
	void bindNode( Node<T>* node )
    {
        Connection<T>* pConn = new Connection<T>( node );
        conns.push_back( pConn );
        node->inConns.push_back( pConn );
    }

    void activate()
    {
        T out = inSum;

        if( !conns.empty() )
        {
            for( int i=0; i < conns.size(); i++ )
            {
                conns[i]->xmit( out );
            }
            printf( " *[%0.3f|%0.3f]", out, grad );
        }

        lastOut = out;

        inSum = (T)0.0;
       
    }

};

template<typename T>
struct Layer
{
    std::vector<Node<T>*> nodes;
    
    Layer<T>* prevLayer;
    Layer<T>* nextLayer;

    int count;

    Layer( int n ) : count(n), prevLayer(NULL), nextLayer(NULL)
    {
        for( int i=0; i < count; i++ )
        {
            nodes.push_back( new Node<T>() );
        }
    }

    void bindLayer( Layer<T>* layer )
    {
        nextLayer = layer;
        nextLayer->prevLayer = this;

        for( int i=0; i<count; i++ )
        {
            for( int j=0; j < layer->count; j++ )
            {
                nodes[i]->bindNode(nextLayer->nodes[j]);
            }
        }
    }

    void calcGradient( T target )
    {
        int i;
        if( nextLayer == NULL ) // output layer
        {
            T outGrad;
            for( i=nodes.size()-1; i>=0; i-- )
            {
                //outGrad = outVal * ( 1.0 - outVal ) * ( target - outVal );
                outGrad =  ( target - nodes[i]->lastOut );
                nodes[i]->grad = outGrad * 1.0; // TODO provided all derivatives here
                printf(" @{%f}\n", nodes[i]->grad );
            }
        }
        else
        {
            for( int n = nodes.size()-1; n >= 0; n-- ) 
            {
                T sum = 0.0;
                for( int c = nodes[n]->conns.size()-1; c >= 0; c-- ) 
                {
                        T nextGrad = nodes[n]->conns[c]->toNode->grad;
                        sum += ( nodes[n]->conns[c]->weight ) * nextGrad;
                }

                nodes[n]->grad = sum;

                printf(" {%f}\n", sum );
            }
        }

        if( prevLayer != NULL )
            //if( prevLayer->prevLayer != NULL )
                prevLayer->calcGradient(target); // target not usedin the following calls
    }

    void updateWeights( T learnRate, T momentum )
    {
        // Update weights
        T alpha, delta, grad, out, weight;
        if( prevLayer != NULL /* || layer == _inLayer */ )
        {
            for( int i=nodes.size()-1; i>=0; i-- )
            {
                for( int c = nodes[i]->inConns.size()-1; c >= 0; c-- ) 
                {
                    alpha = nodes[i]->inConns[c]->alpha;
                    delta = nodes[i]->inConns[c]->delta;
                    grad = nodes[i]->grad;
                    out = nodes[i]->lastOut;
                    //weight = nodes[i]->inConns[c]->weight;

                    //delta = learnRate * grad * out + momentum * delta;
                    delta = momentum * delta + learnRate * grad * out;

                    nodes[i]->inConns[c]->delta = delta;
                    weight = (nodes[i]->inConns[c]->weight += delta); 
                    nodes[i]->inConns[c]->alpha = weight / delta;
                    
                }
            }

            prevLayer->updateWeights( learnRate, momentum );
        }
    }

    T calcError()
    {
        T netErr, outVal;
        int nc = nodes.size();
        for( int i=nc-1; i>=0; i-- )
        {
            outVal = nodes[i]->lastOut;
            netErr +=  ( outVal * outVal );  // TODO: Handle more targets
        }
        netErr /= (T)nc;
        netErr = sqrt( netErr );

        return netErr;
    }

    void activate()
    {
        printf("\n");

        for( int i=0; i<count; i++ )
        {
            nodes[i]->activate();
        }

        if( nextLayer != NULL )
            nextLayer->activate();

        printf("\n");
    }

};


template<typename T>
struct NeuralNet
{

    T _learnRate;
    T _momentum;
    Layer<T> *_inLayer, *_outLayer;

    std::vector<Layer<T>*> layers;

    enum ActType{ linear = 0, sigmoid, tanh };

    ActType _activation;

    NeuralNet( T learn_rate, T momentum, ActType activation )
        : _learnRate( learn_rate ), _momentum( momentum ), _activation( activation )
    {
    }

    void addLayer( int n )
    {
        if( n < 1 )
            return;

        layers.push_back( new Layer<T>(n) );

        int size = layers.size();

        if( size > 1 )
        {
            layers[size-2]->bindLayer( layers[size-1] );
            _outLayer = layers[size-1];
        }
        else
        {
            _inLayer = layers[0];
        }
    }    

    /*
    Layer<T>* getLayer( int n )
    {
        return layers[n];
    } 
    */   

    int getInputNodeCount()
    {
        return _inLayer->nodes.size();
    }

    int getOutputNodeCount()
    {
        return _outLayer->nodes.size();
    }

    void setInput( int inNode, T value )
    {
        _inLayer->nodes[inNode]->input( value );
    }

    T getOutput( int outNode )
    {
        return _outLayer->nodes[outNode]->lastOut;
    }

    void cycle()
    {

        // Start activation recursion
        _inLayer->activate();

    }


    void backProp( T target )
    {

        // * Calc error for layers
        _outLayer->calcError();
        
        // * Calc gradients recursively
        _outLayer->calcGradient( target );

        // Update weights
        _outLayer->updateWeights( _learnRate, _momentum );

        T outVal = _outLayer->nodes[0]->lastOut;

        std::cout << target << "\t" << outVal << "\t" << target-outVal << std::endl;

    }

};


int main( int argc, char**argv)
{

    if( argc < 8 )
    {
        printf("\nusage: ann learn_rate momentum trainings counter beyond in_layer-node-count [follow layers node count]\n");
        printf("\nexample: ./ann 0.002 0.02 15 7 2 1 2 2 1\n\n");

        exit(1);
    }
    
    double lr = atof( argv[1] );
    double mo = atof( argv[2] );
    double trains = atof( argv[3] );
    double end = atof( argv[4] );
    double beyond = atof( argv[5] );

    NeuralNet<double> NN( lr, mo, NeuralNet<double>::linear );

    for( int i=6; i < argc; i++ )
        NN.addLayer( atoi( argv[i] ) );


    double t;
    for( int x=0; x < trains; x++ )
    for( t=1.0; t <= end; t++ )
    {
        NN.setInput( 0, t );

        NN.cycle();

        NN.backProp( t * t );

        //printf( "%f\t", t );

        //std::cout << NN.getOutput( 0 ) << std::endl;
    }

    for( ; t<=end+beyond; t++ )
    {
        NN.setInput( 0, t );

        NN.cycle();

        printf( "%f\t%f\n", t, NN.getOutput( 0 ) );
    }

    return 0;
};




