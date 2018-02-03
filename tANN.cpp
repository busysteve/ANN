
// g++ -g -o ann ANN.cpp XMLTag/xmltag.cpp
// ./ann -w test.weights.xml -r 0.00002 -m 0.0002 -t train.txt -e 10 -i input.txt -l S2 S3 S2 S1
// or
// ./ann -w test.weights.xml -i input.txt

/* train.txt
0 0 0
0 1 1
1 0 1
1 1 0
*/

/* input.txt
0 0
0 1
1 0
1 1
*/

#define dataType double
//#define SAFE( x )    safeguard( x )
#define SAFE( x )    (x)

#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>

#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include "XMLTag/xmltag.h"

//#define log_verbose
//#define log_verbose   printf
#define log_verbose  if( _verbose > 0 ) if( g_counter%_verbose == 0 ) printf

#define log_output  if( _output > 0 ) if( g_counter%_output == 0 ) printf

#define MAX_NN_NAME 30

int _output = 0;
int _verbose = 0;
int g_counter = 0;
int g_threadcount = 0;

//const void* nullptr = NULL;



enum ActType{ linear = 0, sigmoid, tangenth, relu, relul, softMax, none, bias };

template<typename T>
T actNone( T n )
{
	return n;
}


template<typename T>
T actBias( T n )
{
	return (T)1.0;
}


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
T actSoftMax( T n )
{
	return exp(n);
}


template<typename T>
T actTanh( T n )
{
	return tanh( n );
}

template<typename T>
T actReLU( T n )
{
	return (n > 0.0) ? n : 0.0;
}

template<typename T>
T actReLUL( T n )
{
	return (n > 0.0) ? n : (n*.001);
}


template<typename T>
T derivLinear( T n )
{
	return 1.0;
}


template<typename T>
T derivSigmoid( T n )
{
	return n * ( (T)1.0 - n );
}

template<typename T>
T derivSoftMax( T n )
{
	return n * ( (T)1.0 - n );
}


template<typename T>
T derivTanh( T n )
{
	return 1.0 - n * n;
}

template<typename T>
T derivReLU( T n )
{
	return (n < 0.0) ? 0.0 : 1.0;
}

template<typename T>
T derivReLUL( T n )
{
	return 1.0;
	//return (n > 0.0) ? 1.0 : 0.0;
}




//**********************===========================================
//**********************===========================================
//**********************
//**********************===========================================
//**********************===========================================
//**********************
//**********************===========================================
//**********************===========================================
//**********************
//**********************===========================================
//**********************===========================================
//
//=================================================================
//=================================================================
//
//=================================================================
//=================================================================
//
//=================================================================
//=================================================================
//
//
//
//
//

template<typename T>
struct NeuralNet
{

	T _learnRate;
	T _momentum;
    bool _dirty;

    T _lastError;

	char _name[MAX_NN_NAME];

	typedef T ( *derivActFunc )(T);
	typedef T ( *actFunc )(T);

	derivActFunc _derivActFunc;
	actFunc _actFunc;

	bool _bias;
    int  _verbose, _output, _from, _to;

	std::vector< std::vector<T> >               vecLayers;
	std::vector< ActType >                      vecActivationType;
	std::vector< actFunc >                      vecActivationFunc;
	std::vector< derivActFunc >                 vecDerivative;
	std::vector< T >                            vecLayerBias;
    std::vector< std::vector< std::vector<T> > > vecWeights;

	std::vector<T> vecBackPrepTargets;

	NeuralNet( T learn_rate = 0.0001, T momentum = 0.001 )
		: _learnRate( learn_rate ), _momentum( momentum ), _dirty(false)
	{
	}

    ~NeuralNet()
    {
        clear();
    }


	T calcError( std::vector<T> &targets )
	{
		T netErr = (T)0.0, delta;
								 // minus bias
#if 0
        thrust::device_vector<T> dev_targets ( targets.begin(), targets.end() );
        thrust::device_vector<T> dev_lastouts;
    
        
        dev_lastouts.push_back();

        netErr = thrust::reduce( 

		netErr /= (T)nc;
		netErr = sqrt( netErr );


#else
/*
		int nc = nodes.size()-(_bias?1:0);
		for( int i=0; i<nc; i++ )
		{
								 // TODO // handle proper target count!!!!
			//delta = targets[i] - nodes[i]->lastOut;
			delta = nodes[i]->deltaErr = targets[i] - nodes[i]->lastOut;
								 // TODO: Handle more targets
			netErr +=  ( delta * delta ) / 2.0;
            //printf( "%f ", delta * delta );
		}
        log_verbose( "\nsum(netErr)=(%f)\n", netErr );

		netErr /= (T)nc;
		netErr = sqrt( netErr );
*/
#endif
        _lastError = netErr;

        log_output( "\ncalcError(%f)\n", netErr );

		return netErr;
	}

    T lastError()
    {
        return _lastError;
    }


	void calcGradient( std::vector<T> &targets )
	{
/*
		if( nextLayer == NULL )	 // output layer
		{
			T delta;
			//int nc = nodes.size()-(_bias?1:0); // minus bias
								 // minus bias
			int nc = nodes.size();
			for( int i=0; i<nc; i++ )
			{
				//delta =  ( targets[i] - nodes[i]->lastOut );
                delta = 2 * nodes[i]->deltaErr;

			    nodes[i]->grad = delta * _derivActFunc( nodes[i]->lastOut );

				log_verbose("og[%s][%s]outer{delta=%f : last=%f : grad=%f}\n",
					_name, nodes[i]->_name, delta, nodes[i]->lastOut, nodes[i]->grad );
			}
		}
		else
		{
								 // minus bias
			int nc = nodes.size()-(_bias?1:0);
			//int nc = nodes.size(); // minus bias
			for( int n=0; n<nc; n++ )
			{

				T sum = 0.0;
				for( int c = nodes[n]->conns.size()-1; c >= 0; c-- )
				{
					T grad = nodes[n]->conns[c]->toNode->grad;
					sum += ( nodes[n]->conns[c]->weight ) * grad;
					log_verbose("    g[%s][%s]inner{sum=%f:weight=%f:grad=%f}\n",
						_name, nodes[n]->_name, sum, nodes[n]->conns[c]->weight, grad );
				}

                nodes[n]->deltaErr = sum;

				nodes[n]->grad = sum * _derivActFunc( nodes[n]->lastOut );

				log_verbose("ig[%s][%s]inner{sum=%f:sumIn=%f:grad=%f:deriv=%f:out=%f}\n",
					_name, nodes[n]->_name, sum, sumIn, nodes[n]->grad,
					_derivActFunc( nodes[n]->lastOut ), nodes[n]->lastOut );
			}
		}

		if( prevLayer != NULL )
			if( prevLayer->prevLayer != NULL )
								 // target not used in the following calls
				prevLayer->calcGradient(targets);
*/
	}

	void updateWeights( T learnRate, T momentum )
	{
/*
		// Update weights
		T alpha, delta, grad, out, weight;
		T weightSum, weightFactor;
		if( prevLayer != NULL )
		{
			for( int i=nodes.size()-1; i>=0; i-- )
			{

				for( int c = nodes[i]->inConns.size()-1; c >= 0; c-- )
                    weightSum += nodes[i]->inConns[c]->weight;

				for( int c = nodes[i]->inConns.size()-1; c >= 0; c-- )
				{
					Connection<T>* conn = nodes[i]->inConns[c];
					delta = conn->delta;
					grad = nodes[i]->grad;
					//grad = conn->fromNode->grad;
					//out = nodes[i]->lastOut;
					out = conn->fromNode->lastOut;
					weight = conn->weight;

                    //weightFactor = (weight == 0.0 || weightSum == 0.0 ) ? 0.0 : ( nodes[i]->deltaErr * (weightSum / weight) );

					delta = (learnRate * grad * out + momentum * delta); // - weightFactor;

					conn->delta = delta;
					conn->weight += delta;
					log_output("   w[%s][%s]w=%f:w=%f, d=%f, o=%f, g=%f, wf=%f \n",
						_name, conn->_name, weight, conn->weight, delta, out, grad, weightFactor );
				}
			}

			prevLayer->updateWeights( learnRate, momentum );
		}
*/
	}

	void cycle( int dropout_probability_select_mod = 0, T dropout_probability = 1.0 )
	{
		log_verbose("\n");

/*
        if( _activation == softMax )
        {
            sumX = 0.0;
		    for( int i=nodes.size()-1; i>=0; i-- )
            {
                sumX += nodes[i]->_actFunc( nodes[i]->inSum );
            }
        }
*/
        int layerActFuncIdx = 0;
        for( auto &layer : vecLayers )
        {
            actFunc act = vecActivationFunc[layerActFuncIdx];
            //std::transform( layer.begin(), layer.end(), layer.begin(), [=](T x) -> T{ return act(x); } );
            std::transform( layer.begin(), layer.end(), layer.begin(), act );
            layerActFuncIdx++;
        }
/*
        bool dropout_candidate_select = false;

        if( dropout_probability < 1.0 && dropout_probability_select_mod != 0 )
            dropout_candidate_select = ( (rand()%dropout_probability_select_mod) == 0 );
*/

        

		log_verbose("\n");
	}

	void setLearnRate( T lr )
	{
		_learnRate = lr;
	}

	void setMomentum( T mo )
	{
		_momentum = mo;
	}

	void clear()
	{

	}

	int addLayer( int n, ActType act, bool bias )
	{
		if( n < 1 )
			return 0;

		char name[MAX_NN_NAME];

		//sprintf( name, "L%d", (int)layers.size() );


        switch( act )
        {
		    case linear:
			    _actFunc = actLinear<T>;
			    _derivActFunc = derivLinear<T>;
		        break;
		    case sigmoid:
			    _actFunc = actSigmoid<T>;
			    _derivActFunc = derivSigmoid<T>;
                break;
		    case tangenth:
			    _actFunc = actTanh<T>;
			    _derivActFunc = derivTanh<T>;
                break;
		    case softMax:
			    _actFunc = actSoftMax<T>;
			    _derivActFunc = derivSoftMax<T>;
                break;
		    case relu:
			    _actFunc = actReLU<T>;
			    _derivActFunc = derivReLU<T>;
		        break;
		    case relul:
		    {
			    _actFunc = actReLUL<T>;
			    _derivActFunc = derivReLUL<T>;
		    }
            default:
                return(0);
        }

		vecLayers.push_back( std::vector<T>(n) );

        return n;
	}

	std::vector<T>& getLayer( int n )
	{
		return vecLayers[n];
	}

	int getInputNodeCount()
	{
        int s = vecLayers.size();
		if( s > 0 )
			return vecLayers[0].size();

		return 0;
	}

	int getOutputNodeCount()
	{
        int s = vecLayers.size();
		if( s > 0 )
			return vecLayers[s-1].size();

		return 0;
	}

	void setInput( int inNode, T value )
	{
        if( !vecLayers.empty() && inNode < vecLayers[0].size() )
			vecLayers[0][inNode] = value;
	}

// HERE!!!!!!!!!!!!!!

	T getOutput( int outNode )
	{
        int s = vecLayers.size();
        if( s>0 && outNode < vecLayers[s-1].size() )
			return vecLayers[s-1][outNode];

		return 0;
	}


	void backPushTargets( T t )
	{
		vecBackPrepTargets.push_back( t );
	}

	T backPropagate()
	{
        _dirty = true;

		// * Calc error for layers
		T layer_error = calcError( vecBackPrepTargets );

		// * Calc gradients recursively
		calcGradient( vecBackPrepTargets );

		// Update weights recursively
		updateWeights( _learnRate, _momentum );

		//T outVal = _outLayer->nodes[0]->lastOut;

		vecBackPrepTargets.clear();

        return layer_error;
	}

	void store( std::string fileName )
	{
/*
        if( !_dirty )
            return;

		XMLTag xml("NeuralNet");

		Layer<dataType>* layer = _inLayer;

		while( layer->nextLayer != NULL || layer == _outLayer )
		{
			XMLTag &refLayer = xml.addTag( "layer" );

			if( layer == _inLayer )
				refLayer.setAttribute( "name", "input_layer" );
			else if( layer == _outLayer )
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
			else if( act == softMax )
			{
				activation = "softMax";
			}
			else if( act == tangenth )
			{
				activation = "tangenth";
			}
			else if( act == relu )
			{
				activation = "relu";
			}
			else if( act == relul )
			{
				activation = "relul";
			}
			else if( act == none )
			{
				activation = "none";
			}

			refLayer.setAttribute( "activation", activation );
			refLayer.setAttribute( "bias", layer->_bias );

			XMLTag &refNodes = refLayer.addTag( "nodes" );

			for( int n=0; n < layer->nodes.size(); n++ )
			{
				XMLTag &refNode = refNodes.addTag( "node" );

				refNode.setAttribute( "bias", layer->nodes[n]->_bias );

				if( layer != _outLayer )
				{
					XMLTag &refConnections = refNode.addTag( "connections" );

					for( int c=0; c < layer->nodes[n]->conns.size(); c++ )
					{
						XMLTag &refConnection = refConnections.addTag( "connection" );
						refConnection.addTag( "weight", layer->nodes[n]->conns[c]->weight );
					}
				}
			}

			if( layer == _outLayer )
				break;

			layer = layer->nextLayer;
		}

		xml.store( fileName.c_str(), true );

        _dirty = false;
*/
	}

	void load( std::string fileName )
	{
/*
		XMLTag NNxml;

		NNxml.load( fileName.c_str() );

		clear();

		for( int layer = 0; layer < NNxml.count(); layer++ )
		{
			std::string activation = NNxml[layer].attribute( "activation" );
			bool bias = NNxml[layer].boolAttribute( "bias" );

			XMLTag &xNodes = NNxml[layer]["nodes"];
			Layer<T> *pLayer = NULL;

			int count = xNodes.count();

			if( bias )  count--;

			// Add Layer - with nodes
			if( activation == "linear" )
				pLayer = addLayer( count, linear, bias );
			else if( activation == "sigmoid" )
				pLayer = addLayer( count, sigmoid, bias );
			else if( activation == "tangenth" )
				pLayer = addLayer( count, tangenth, bias );
			else if( activation == "relu" ) 
				pLayer = addLayer( count, relu, bias );
			else if( activation == "relul" ) 
				pLayer = addLayer( count, relul, bias );
			else if( activation == "softMax" ) 
				pLayer = addLayer( count, softMax, bias );
		}

		try
		{
			for( int layer = 0; layer < NNxml.count(); layer++ )
			{

				XMLTag &xNodes = NNxml[layer]["nodes"];

				for( int node=0; node<xNodes.count(); node++ )
				{
					XMLTag &xConnections = xNodes[node]["connections"];

					for( int conn=0; conn<xConnections.count(); conn++ )
						layers[layer]->nodes[node]->conns[conn]->weight = xConnections[conn]["weight"].floatValue();
				}
			}
		}
		catch(...){}

        _dirty = false;
*/
	}
};


//======================================================================================
//======================================================================================
//======================================================================================
//======================================================================================
//======================================================================================
//======================================================================================
//======================================================================================
//======================================================================================
//======================================================================================
//======================================================================================
//======================================================================================
//======================================================================================
//======================================================================================
//======================================================================================
//======================================================================================
//======================================================================================
//======================================================================================
//======================================================================================
//======================================================================================




int main( int argc, char**argv)
{

	if( argc < 3 )
	{
		printf("\nusage: ann -w [(r/w)weights (restore) file name] [-i input file ] { -t training_file -x training_iterations -r learn_rate -m momentum -l [Layer spec] }\n");
		printf("\nexample: ./ann -w test.weights.xml -r 0.00002 -m 0.0002 -t train.txt -x 10 -i input.txt -l S2 S3 S2 S1\n\n");
		printf( "Layer types must be L, S, T, R, or e prefixed to the Node count.\n" );

		exit(1);
	}

	srand( time(NULL) );

	std::string strTrainingFile, strInputFile, strWeights ( "temp.weights.xml" );
	dataType lr=0.0, mo=0.0;
	int i = 1, training_iterations=1, times=1;
	FILE *t_fp = NULL, *i_fp = NULL;
	bool bias = false;
	bool cont = false;
    bool bDisplayErrors = false;
    int  store_every_time = 0;
    bool one_or_zero = false;
    dataType errorStopLearning = 0.0;
    dataType noDropProbability = 1.0;
    int noDropSelectMod = 0;
	NeuralNet<dataType> NN;
    int trainingTestOutputMod = 0;

	while( i < argc && argv[i][0] == '-' )
	{
		switch( argv[i][1] )
		{
			case '1':
				++i;
                one_or_zero = true;
				break;
			case 'E':
			case 'S':
				++i;
				errorStopLearning = atof(argv[i]);
				++i;
				break;
			case 'D':
				++i;
				noDropProbability = atof(argv[i]);
				++i;
				break;
			case 'M':
				++i;
				noDropSelectMod = atoi(argv[i]);
				++i;
				break;
			case 'O':
				++i;
				trainingTestOutputMod = atoi(argv[i]);
				++i;
				break;
			case 'W':
				++i;
				store_every_time = atoi(argv[i]);
				++i;
			case 'w':
				++i;
				strWeights = argv[i];
				++i;
				break;
			case 'i':
				++i;
				strInputFile = argv[i];
				++i;
				if( strInputFile == "-" )
					i_fp = stdin;
				else				
					i_fp = fopen( strInputFile.c_str(), "r+" );
				break;
			case 'b':
				++i;
				bias = true;
				break;
			case 'v':
				++i;
				_verbose = 1;
				if( argv[i][0] != '-' )
				{
					_verbose = atoi( argv[i] );
				}
				++i;
				break;
			case 'T':
				++i;
				g_threadcount = 1;
				if( argv[i][0] != '-' )
				{
					g_threadcount = atoi( argv[i] );
				}
				++i;
				break;
			case 'o':
				++i;
				_output = 1;
				if( argv[i][0] != '-' )
				{
					_output = atoi( argv[i] );
				}
				++i;
				break;
			case 'c':
				++i;
				cont = true;
				break;
			case 't':
				++i;
				strTrainingFile = argv[i];
				++i;
                if( strTrainingFile == "-" )
                    t_fp = stdin;
                else
				    t_fp = fopen( strTrainingFile.c_str(), "r+" );
				break;
			case 'e':
				++i;
				training_iterations = atoi( argv[i] );
				++i;
				break;
			case 'x':
				++i;
				times = atoi( argv[i] );
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
                        int verbose = 0;
                        int output = 0;

                        int b=0;
                        bool bias = false;

                        if( argv[i][b] == 'b' )
                            b=1; 
                        if( b == 1 )
                            bias = true;

                        int from = 0, to = 0;;

                        int x = 0;
                        int len=0;

                        for( len=x=strlen(argv[i]); x > 0; x-- )
                        {
                            if( argv[i][x] == '@' )
                            { 
                                to = from = x+1;
                                argv[i][x] = '\0';
                                break;
                            }
                        }

                        for( x=len; x > from; x-- )
                        {
                            if( argv[i][x] == '-' )
                            { 
                                to = x+1;
                                argv[i][x] = '\0';
                                break;
                            }
                        }

                        if( to >= from )
                        {
                            to = atoi(&argv[i][to]);
                            //printf(" to=%d ", to );
                        }

                        if( from > 0 )
                        {
                            from = atoi(&argv[i][from]);
                            //printf(" from=%d ", from );
                        }

                        //printf("\n\n ****** %s ****** \n\n", &argv[i][b] );

                        if( from > 0 || to > 0 )
                        {
                            verbose = _verbose;
                            output = _output;
                        }

						switch( argv[i][b++] )
						{
							case 'L':
								NN.addLayer( atoi( &argv[i][b] ), linear, bias );
								break;
							case 'S':
								NN.addLayer( atoi( &argv[i][b] ), sigmoid, bias );
								break;
							case 'T':
								NN.addLayer( atoi( &argv[i][b] ), tangenth, bias );
								break;
							case 'R':
								NN.addLayer( atoi( &argv[i][b] ), relu, bias );
								break;
							case 'r':
								NN.addLayer( atoi( &argv[i][b] ), relul, bias );
								break;
							case 'X':
								NN.addLayer( atoi( &argv[i][b] ), softMax, bias );
								break;
							case 'N':
								NN.addLayer( atoi( &argv[i][b] ), none, bias );
								break;
							case '-':
								break;
							default:
								printf( "Layer types must be L, S, R, r, X or T prefixed to the Node count.\n" );
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



	if( cont == true )
	{
		NN.load( strWeights.c_str() );
	}

	if( t_fp != NULL )			 // training file
	{

		//printf("\n");


        unsigned long counter=0;


		int ic = NN.getInputNodeCount();
		int oc = NN.getOutputNodeCount();
		ssize_t read;
		char *pch;

		//char tmpline[1024];

        for( int e=0; e < training_iterations; e++ )
        {
		    fseek( t_fp, 0, SEEK_SET );

		    char *line = NULL;
		    size_t len = 0;
            dataType lastError, runningError = 0.0;

		    while( (read = getline(&line, &len, t_fp)) != -1 )
		    {

			    // Cycle inputs
			    //memcpy( tmpline, line, len+1 );

	            for( int x = 0; x < times; x++ )
	            {
			        dataType val;
			        pch = strtok (line," \t,:");
			        for( int t=0; (t < ic) && (pch != NULL); t++ )
			        {
				        sscanf (pch, "%lf\n",&val);
				        pch = strtok (NULL, " \t,:");
				        NN.setInput( t, val );
				        log_output( "I%d=%lf ", t, val );
			        }
			        NN.cycle(noDropSelectMod, noDropProbability);

                    if( trainingTestOutputMod > 0 )
                    {
                        if( (counter % trainingTestOutputMod) == 0 )
                        {
                            printf("\n");
			                // Set targets for back propagation (training)
			                for( int t=0; (t < oc); t++ )
			                {
				                log_output( "o%d=", t );
                                if( one_or_zero == false )
				                    printf( "%f ", NN.getOutput(t) );
                                else
                                    printf( "%d ", NN.getOutput(t) >= 0.5 ? 1 : 0 );

			                }

                            printf("\n");
			                fflush( stdout );
                        }
                    }

			        // Set targets for back propagation (training)
			        for( int t=0; (t < oc) && (pch != NULL); t++ )
			        {
				        sscanf (pch, "%lf\n",&val);
				        pch = strtok (NULL, " \t,:");
				        NN.backPushTargets( val );
				        log_output( "O%d=%f ", t, val );
			        }

			        runningError += lastError = NN.backPropagate( );

			        log_output( "[%f]<%f>", NN.getOutput(0), val - NN.getOutput(0) );

			        log_output( "\n" );


	            	printf("\r%1.12f  %1.12f %d", runningError / (counter+1), lastError, x+1 );
	            	//fflush( stdout );



	            	printf(" %lu ", ++counter );
	            	fflush( stdout );

                    g_counter++;
                }
		    }

            free( line );

   			printf("   %d epochs            ", e+1 );
            fflush( stdout );

            if(  store_every_time != 0 && ( (e+1) % store_every_time == 0) )
            {
                if( store_every_time < 0 )
                {
                    char epch[255];
                    sprintf( epch, "-%06d", e+1 );
                    NN.store( (strWeights+epch).c_str() );
                    printf("  -  Stored Weights in %s\n", (strWeights+epch).c_str() );
                }
                else
                {
                    NN.store( strWeights.c_str() );
                    printf("  -  Stored Weights in %s\n", strWeights.c_str() );
                }
            }

            if( errorStopLearning > 0.0 )
                if( lastError <= errorStopLearning )
                    break;
        }

        NN.store( strWeights.c_str() );

        printf("\n");

	//	if( cont != true )
			

	}
	else
	{
		NN.load( strWeights.c_str() );
	}

	if( i_fp != NULL )			 // input file
	{
		//fseek( i_fp, 0, SEEK_SET );

		int ic = NN.getInputNodeCount();
		int oc = NN.getOutputNodeCount();

		char *line = NULL;
		size_t len = 0;
		ssize_t read;
		char *pch;

		//if( g_output > 0 ) g_output = 1;

		while( (read = getline(&line, &len, i_fp)) != -1 )
		{
			// Cycle inputs
			dataType val;
			pch = strtok (line," \t,:");
			for( int t=0; (t < ic) && (pch != NULL); t++ )
			{
				sscanf (pch, "%lf\n",&val);
				pch = strtok (NULL, " \t,:");
				NN.setInput( t, val );
				log_output( "i%d=%f ", t, val );
			}
			NN.cycle();


			// Set targets for back propagation (training)
			for( int t=0; (t < oc); t++ )
			{
				log_output( "o%d=", t );
                if( one_or_zero == false )
				    printf( "%f ", NN.getOutput(t) );
                else
                    printf( "%d ", NN.getOutput(t) >= 0.5 ? 1 : 0 );

				fflush( stdout );
			}

			//printf( "[%f]", NN.getOutput(0) );

			printf( "\n" );
		}

	}
                


	return 0;
};
