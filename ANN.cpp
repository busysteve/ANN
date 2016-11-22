
// g++ -g -o ann ANN.cpp

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>

const void* nullptr = NULL;

enum ActType{ linear = 0, sigmoid, tangenth, cubed, naturalLog };


template<typename T>
T actLinear( T n )
{
    return n;
}

template<typename T>
T actSigmoid( T n )
{
    return 1.0 / ( 1.0 + exp(-n) );
}

template<typename T>
T actTanh( T n )
{
    return tanh( n );
}

template<typename T>
T actCubed( T n )
{
    return ( n * n * n );
}

template<typename T>
T actLn( T n )
{
    return log( n );
}

template<typename T>
T derivLinear( T n )
{
    return 1.0;
}

template<typename T>
T derivSigmoid( T n )
{
    return n * ( 1.0 - n );
}

template<typename T>
T derivTanh( T n )
{
    return 1.0 - n * n;
}

template<typename T>
T derivCubed( T n )
{
    return 3 * ( n * n );
}

template<typename T>
T derivLn( T n )
{
    return log( n );
}






template<typename T>
struct Node;

#define log //printf

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
            log( " <%0.3f|%0.3f>(%f)\n", in, weight, in*weight );
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
    
    //ActType _activation;

    typedef T ( *ActFunc )(T);

    ActFunc _actFunc;

    Node( ActFunc actFunc ) : 
        inSum((T)0.0), lastOut((T)0.0), 
        deltaErr((T)0.0), grad((T)1.0), 
        _actFunc(actFunc) {}

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
                conns[i]->xmit( _actFunc( out ) );
            }
            log( " *[%0.3f|%0.3f]", out, grad );
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

    ActType _activation;

    typedef T ( *derivActFunc )(T);

    derivActFunc _derivActFunc;

    int count;

    Layer( int n, ActType act ) : count(n), prevLayer(NULL), nextLayer(NULL), _activation(act)
    {
        for( int i=0; i < count; i++ )
        {
            if( act == linear )
            {
                nodes.push_back( new Node<T>( actLinear<T> ) );
                _derivActFunc = derivLinear<T>;
            }
            else if( act == sigmoid )
            {
                nodes.push_back( new Node<T>( actSigmoid<T> ) );
                _derivActFunc = derivSigmoid<T>;
            }
            else if( act == tangenth )
            {
                nodes.push_back( new Node<T>( actTanh<T> ) );
                _derivActFunc = derivTanh<T>;
            }            
            else if( act == cubed )
            {
                nodes.push_back( new Node<T>( actCubed<T> ) );
                _derivActFunc = derivCubed<T>;
            }            
            else if( act == naturalLog )
            {
                nodes.push_back( new Node<T>( actLn<T> ) );
                _derivActFunc = derivLn<T>;
            }            
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
                outGrad =  ( target - nodes[i]->lastOut );
                nodes[i]->grad = outGrad * _derivActFunc( target );
                log(" @{%f}\n", nodes[i]->grad );
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

                log(" {%f}\n", sum );
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
        log("\n");

        for( int i=0; i<count; i++ )
        {
            nodes[i]->activate();
        }

        if( nextLayer != NULL )
            nextLayer->activate();

        log("\n");
    }

};


template<typename T>
struct NeuralNet
{

    T _learnRate;
    T _momentum;
    Layer<T> *_inLayer, *_outLayer;

    std::vector<Layer<T>*> layers;


    NeuralNet( T learn_rate, T momentum )
        : _learnRate( learn_rate ), _momentum( momentum )
    {
    }

    void addLayer( int n, ActType act )
    {
        if( n < 1 )
            return;

        layers.push_back( new Layer<T>(n, act ) );

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

    }

};


double problem( double n )
{
    return n*n;
}

int main( int argc, char**argv)
{

    if( argc < 8 )
    {
        printf("\nusage: ann learn_rate momentum trainings counter beyond in_layer-node-count [follow layers node count]\n");
        printf("\nexample: ./ann 0.002 0.02 15 7 2 T1 T2 T2 L1\n\n");

        exit(1);
    }
    
    double lr = atof( argv[1] );
    double mo = atof( argv[2] );
    double trains = atof( argv[3] );
    double end = atof( argv[4] );
    double beyond = atof( argv[5] );

    NeuralNet<double> NN( lr, mo );

    for( int i=6; i < argc; i++ )
    {
        switch( argv[i][0] )
        {
            case 'L':
                NN.addLayer( atoi( &argv[i][1] ), linear );
                break;
            case 'S':
                NN.addLayer( atoi( &argv[i][1] ), sigmoid );
                break;
            case 'T':
                NN.addLayer( atoi( &argv[i][1] ), tangenth );
                break;
            case 'C':
                NN.addLayer( atoi( &argv[i][1] ), cubed );
                break;
            case 'e':
                NN.addLayer( atoi( &argv[i][1] ), cubed );
                break;
            default:
                printf( "Layer types must be L, S, T, C, or e prefixed to the Node count.\n" );
                exit(1);
        }
    }

    double t, out;
    for( int x=1; x <= trains; x++ )
    for( t=1.0; t <= end; t++ )
    {
        NN.setInput( 0, t );

        NN.cycle();

        NN.backProp( problem(t) );

        out = NN.getOutput( 0 );

        if( x == trains )
            printf( "%f\t%f\t%f\n", t, out, problem(t) - out);
        
    }

    for( ; t<=end+beyond; t++ )
    {
        NN.setInput( 0, t );

        NN.cycle();

        out = NN.getOutput( 0 );

        printf( "%f\t%f\t%f\t%f\n", t, out, problem(t) - out, problem(t) );
    }

    return 0;
};



