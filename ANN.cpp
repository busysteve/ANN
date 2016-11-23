
// g++ -g -o ann ANN.cpp

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include "XMLTag/xmltag.h"


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
struct Node;

#define log printf

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

    void loadweight( FILE *fp )
    {
        fscanf( fp, "%lf", &weight );
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

    Layer( int n, ActType act ) 
        : count(n), prevLayer(NULL), nextLayer(NULL), _activation(act)
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

    

    void calcGradient( std::vector<T> &targets )
    {
        int i;
        if( nextLayer == NULL ) // output layer
        {
            T outGrad;
            for( i=nodes.size()-1; i>=0; i-- )
            {
                outGrad =  ( targets[i] - nodes[i]->lastOut );
                nodes[i]->grad = outGrad * _derivActFunc( targets[i] );
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
                prevLayer->calcGradient(targets); // target not usedin the following calls
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

    std::vector<T> vecBackPrepTargets;

    NeuralNet( T learn_rate = 0.0001, T momentum = 0.001 )
        : _learnRate( learn_rate ), _momentum( momentum )
    {
    }

	void setLearnRate( T lr )
	{
		_learnRate = lr;
	}
	
	void setMomentum( T mo )
	{
		_momentum = mo;
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

    
    Layer<T>* getLayer( int n )
    {
        return layers[n];
    } 
     

    int getInputNodeCount()
    {
		if( _inLayer != NULL )
			return _inLayer->nodes.size();
		return 0;
    }

    int getOutputNodeCount()
    {
		if( _outLayer != NULL )
			return _outLayer->nodes.size();
		return 0;
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


    void backPushTargets( T t )
    {
        vecBackPrepTargets.push_back( t );
    }

    void backPropagate()
    {

        // * Calc error for layers
        _outLayer->calcError();
        
        // * Calc gradients recursively
        _outLayer->calcGradient( vecBackPrepTargets );

        // Update weights
        _outLayer->updateWeights( _learnRate, _momentum );

        //T outVal = _outLayer->nodes[0]->lastOut;
        
        vecBackPrepTargets.clear();

    }

	void store( std::string fileName )
	{
		XMLTag xml("NeuralNet");
		
		Layer<double>* layer = _inLayer;

		while( layer->nextLayer != NULL )
		{
			XMLTag &refLayer = xml.addTag( "layer" );
			
			if( layer == _inLayer )
				refLayer.setAttribute( "name", "input_layer" );
			else if( layer == _inLayer )
				refLayer.setAttribute( "name", "output_layer" );
			else
				refLayer.setAttribute( "name", "hidden_layer" );
			std::string activation;
			
			ActType act = layer->_activation;
			
            if( act == linear )
            {
				activation = "linear";
            }
            else if( act == sigmoid )
            {
				activation = "sigmoid";
            }
            else if( act == tangenth )
            {
				activation = "tangenth";
            }
            
			refLayer.setAttribute( "activation", activation );
						
			XMLTag &refNodes = refLayer.addTag( "nodes" );
			
			for( int n=0; n < _inLayer->nodes.size(); n++ )
			{
				XMLTag &refNode = refNodes.addTag( "node" );
				
				XMLTag &refConnections = refNode.addTag( "connections" );

				for( int c=0; c < _inLayer->nodes[n]->conns.size(); c++ )
				{
					XMLTag &refConnection = refConnections.addTag( "connection" );
					refConnection.addTag( "weight", _inLayer->nodes[n]->conns[c]->weight );
				}
			}
			
			layer = layer->nextLayer;
		}
		
		xml.store( fileName.c_str() );
	}
};


int main( int argc, char**argv)
{

    if( argc < 3 )
    {
		printf("\nusage: ann -w [(r/w)weights (restore) file name] { -t training_file -r learn_rate -m momentum -l [Layer spec] }\n");
        printf("\nexample: ./ann -w saved.weights.xml -r 0.002 -m 0.02 -l S2 S3 S2 L1\n\n");
		printf( "Layer types must be L, S, T, C, or e prefixed to the Node count.\n" );	

        exit(1);
    }

	srand( time(NULL) );
    
	std::string strTrainingFile, strWeights ( "temp.weights.xml" );
	double lr=0.0, mo=0.0;
    int i = 1, training_iterations=1;
	FILE *t_fp = NULL;

    NeuralNet<double> NN;

    while( i < argc && argv[i][0] == '-' )
    {
        switch( argv[i][1] )
        {
            case 'w':
                ++i;
                strWeights = argv[i];
				++i;
                break;
            case 't':
                ++i;
                strTrainingFile = argv[i];
				++i;
				t_fp = fopen( strTrainingFile.c_str(), "r" );
                break;
            case 'i':
                ++i;
                training_iterations = atoi( argv[i] );
				++i;
                break;
            case 'r':
                ++i;
                lr = atof( argv[i] );
				NN.setLearnRate( lr );
				++i;
                break;
            case 'm':
                ++i;
                mo = atof( argv[i] );
				NN.setMomentum( mo );
				++i;
                break;
            case 's':
                ++i;
                srand( atoi( argv[i] ) );
				++i;
                break;
            case 'l':
				++i;
				{
					for( ; i < argc; i++ )
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
							case '-':
								break;
							default:
								printf( "Layer types must be L, S, T, C, or e prefixed to the Node count.\n" );
								exit(1);
						}
					}
					break;
				}
            default:
				printf("Unknown switch ( -%c )\n", argv[i][1] );
				exit(1);
        }
		
    }
	
	int ic = NN.getInputNodeCount();
	int oc = NN.getOutputNodeCount();
	
	
	if( t_fp != NULL )
	{
		
		for( int x=0; x < training_iterations; x++ )
		{
			fseek( t_fp, 0, SEEK_SET );
			
			while( !feof(t_fp) )
			{
				// Cycle inputs
				for( int t=0; t < ic; t++ )
				{
					double val;	
					fscanf( t_fp, "%lf", &val );
					NN.setInput( t, val );
					
					printf( "I%d=%lf ", t, val );
				}
				NN.cycle();

				// Set targets for back propagation (training)
				for( int t=0; t < oc; t++ )
				{
					double val;	
					fscanf( t_fp, "%lf", &val );
					printf( "O%d=%lf ", t, val );
					NN.backPushTargets( val );  
					

				}
				NN.backPropagate();
				
				printf( "\n" );
			}
		}
		
		NN.store( strWeights.c_str() );
				
	}
	
    return 0;
};


