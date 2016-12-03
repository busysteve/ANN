
// g++ -g -o ann ANN.cpp XMLTag/xmltag.cpp
// ./ann -w test.weights.xml -r 0.00002 -m 0.0002 -t train.txt -x 10 -i input.txt -l S2 S3 S2 S1
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
#define log_verbose  if( g_verbose > 0 ) if( g_counter%g_verbose == 0 ) printf

#define log_output  if( g_output > 0 ) if( g_counter%g_output == 0 ) printf

#define MAX_NN_NAME 30

int g_output = 0;
int g_verbose = 0;
int g_counter = 0;

const void* nullptr = NULL;

enum ActType{ linear = 0, sigmoid, tangenth, none, bias };

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
T safeguard( T n )
{
	return n;
	//return n != 0.0 ? n : 0.0000000000001;
}


template<typename T>
struct Layer;

template<typename T>
struct Node;

template<typename T>
struct Connection
{

	T weight, alpha, delta;

	Node<T> *fromNode, *toNode;

	char _name[MAX_NN_NAME];

	Connection( Node<T>* fNode, Node<T>* tNode )
		: fromNode( fNode ), toNode( tNode ), alpha((T)1.0), delta((T)0.0)
	{

		sprintf( _name, "C-%s-%s", fromNode->_name, toNode->_name );

		T rnd = (T)std::rand() / RAND_MAX;

		weight = rnd + 0.000000001;
	}

	void xmit( T in )
	{
		if( toNode != nullptr )
		{
								 // Apply weight here
			toNode->in( in * weight );
			log_verbose( "xm[%s]<in=%0.3f|w=%0.3f>(%0.3f)\n", _name, in, weight, in*weight );
		}
	}

};

template<typename T>
struct Node
{

	T inSum, lastOut;
	T deltaErr;
	T grad;
	bool _bias;

	char _name[MAX_NN_NAME];

	std::vector<Connection<T>*> conns;
	std::vector<Connection<T>*> inConns;

	bool _activate;

	//ActType _activation;

	typedef T ( *ActFunc )(T);

	ActFunc _actFunc;

	Node( ActFunc actFunc, bool bias, char* name ) :
	inSum((T)0.0), lastOut((T)0.0),
		deltaErr((T)0.0), grad((T)1.0),
		_actFunc(actFunc), _bias(bias), _activate(false)
	{
		sprintf(_name, "%s", name );
	}

	void input( T in )
	{
		inSum += in;			 // Sum weighted inputs for activation
		log_verbose( "in[%s]{%0.3f}SUM(%0.3f)\n", _name, in, inSum );
	}

	void in( T in )
	{
		_activate = true;
		inSum += in;			 // Sum weighted inputs for activation
		log_verbose( "in[%s]{%0.3f}SUM(%0.3f)\n", _name, in, inSum );
	}

	T out()
	{
		return ( lastOut );
	}

	// Node to bind to (next layer node)
	void bindNode( Node<T>* node )
	{
		Connection<T>* pConn = new Connection<T>( this, node );
		conns.push_back( pConn );
		node->inConns.push_back( pConn );

	}

	T cycle( )
	{
		if( _activate || _bias)
			lastOut = _actFunc( inSum );
		else
			lastOut = inSum;

		if( !conns.empty() )
		{
			for( int i=0; i < conns.size(); i++ )
			{
				conns[i]->xmit( lastOut );
			}
		}

		if( _bias )
			log_verbose( "[%s](in=%lf)--------- bias --------(out=%lf)\n", _name, inSum, lastOut );
		else
			log_verbose( "[%s](in=%lf)-----------------------(out=%lf)\n", _name, inSum, lastOut );

		inSum = (T)0.0;

		return lastOut;
	}

};

template<typename T>
struct Layer
{
	std::vector<Node<T>*> nodes;

	Layer<T>* prevLayer;
	Layer<T>* nextLayer;

	char _name[MAX_NN_NAME];

	ActType _activation;

	typedef T ( *derivActFunc )(T);
	typedef T ( *actFunc )(T);

	derivActFunc _derivActFunc;
	actFunc _actFunc;

	bool _bias;

	int count;

	T sumIn;

	Layer( int n, ActType act, bool bias, char* name )
		: count(n), prevLayer(NULL), nextLayer(NULL), _activation(act), _bias(bias),
		sumIn(0.0)
	{

		strcpy( _name, name );

		if( act == linear )
		{
			_actFunc = actLinear<T>;
			_derivActFunc = derivLinear<T>;
		}
		else if( act == sigmoid )
		{
			_actFunc = actSigmoid<T>;
			_derivActFunc = derivSigmoid<T>;
		}
		else if( act == tangenth )
		{
			_actFunc = actTanh<T>;
			_derivActFunc = derivTanh<T>;
		}
		else if( act == none )
		{
			_actFunc = actNone<T>;
			_derivActFunc = actNone<T>;
		}

		for( int i=0; i < count; i++ )
		{
			char tmpname[MAX_NN_NAME];
			sprintf(tmpname, "N%d-%s", (int)nodes.size(), _name );
			nodes.push_back( new Node<T>( _actFunc, false, tmpname ) );
		}

		if( bias == true )
		{
			char tmpname[MAX_NN_NAME];
			sprintf(tmpname, "B%d-%s", (int)nodes.size(), _name );
			//nodes.push_back( new Node<T>( _actFunc, true, tmpname ) );
			nodes.push_back( new Node<T>( actBias<T>, true, tmpname ) );
			//_derivActFunc = actBias<T>;
			//_derivActFunc = actBias<T>;
		}
	}

	void bindLayer( Layer<T>* layer )
	{
		nextLayer = layer;
		nextLayer->prevLayer = this;

		for( int i=0; i<nodes.size(); i++ )
		{
			for( int j=0; j < layer->count; j++ )
			{
				nodes[i]->bindNode(nextLayer->nodes[j]);
			}
		}
	}

	T calcError( std::vector<T> &targets )
	{
		T netErr = (T)0.0, delta;
								 // minus bias
		int nc = nodes.size()-(_bias?1:0);
		for( int i=0; i<nc; i++ )
		{
								 // TODO // handle proper target count!!!!
			delta = targets[i] - nodes[i]->lastOut;
								 // TODO: Handle more targets
			netErr +=  ( delta * delta );
		}
		netErr /= (T)nc;
		netErr = sqrt( netErr );

		return netErr;
	}

	T sumDOW( Layer<T> *nLayer )
	{
		T sum = 0.0;
		int ns = nLayer->nodes.size()-( nLayer->_bias ? 1 : 0 );

		for( int n = 0; n < ns; n++ )
		{
			for( int c = 0; c < nLayer->nodes[n]->conns.size(); c++ )
			{
				T grad = nLayer->nodes[n]->conns[c]->toNode->grad;
				sum += ( nLayer->nodes[n]->conns[c]->weight ) * grad;

				log_verbose("    sumDOW[%s][%s]inner{sum=%f:weight=%f:grad=%f}\n",
					nLayer->_name, nLayer->nodes[n]->_name, sum,
					nLayer->nodes[n]->conns[c]->weight, grad );
			}
		}

		return sum;
	}

	void calcGradient( std::vector<T> &targets )
	{
		if( nextLayer == NULL )	 // output layer
		{
			T delta;
			//int nc = nodes.size()-(_bias?1:0); // minus bias
								 // minus bias
			int nc = nodes.size();
			for( int i=0; i<nc; i++ )
			{
				delta =  ( targets[i] - nodes[i]->lastOut );

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

				//T sum = sumDOW( nextLayer );
				//T sum = sumDOW( this );

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
					Connection<T>* conn = nodes[i]->inConns[c];
					delta = conn->delta;
					grad = nodes[i]->grad;
					//grad = conn->fromNode->grad;
					//out = nodes[i]->lastOut;
					out = conn->fromNode->lastOut;
					weight = conn->weight;

					delta = learnRate * grad * out + momentum * delta;

					conn->delta = delta;
					conn->weight += delta;
					log_verbose("   w[%s][%s]w=%lf:w=%lf, d=%lf, o=%lf, g=%lf \n",
						_name, conn->_name, weight, conn->weight, delta, out, grad );
				}
			}

			prevLayer->updateWeights( learnRate, momentum );
		}
	}

	void cycle(  )
	{
		log_verbose("\n");

		sumIn = 0.0;

		for( int i=nodes.size()-1; i>=0; i-- )
			sumIn += nodes[i]->cycle();

		if( nextLayer != NULL )
			nextLayer->cycle( );

		log_verbose("\n");
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

	void clear()
	{
		for( int l=0; l<layers.size(); l++ )
		{
			Layer<T> *pLayer = layers[l];
			for( int n=0; n<pLayer->nodes.size(); n++ )
			{
				Node<T> *node = pLayer->nodes[n];
				for( int c=0; c<node->conns.size(); c++ )
				{
					delete node->conns[c];
				}
				delete node;
			}
			delete pLayer;
		}
		layers.clear();
	}

	Layer<T>* addLayer( int n, ActType act, bool bias )
	{
		if( n < 1 )
			return NULL;

		Layer<T>* pl;

		char name[MAX_NN_NAME];

		sprintf( name, "L%d", (int)layers.size() );

		layers.push_back( pl = new Layer<T>(n, act, bias, name ) );

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

		return pl;
	}

	Layer<T>* getLayer( int n )
	{
		return layers[n];
	}

	int getInputNodeCount()
	{
		if( _inLayer != NULL )
			return _inLayer->count;
		return 0;
	}

	int getOutputNodeCount()
	{
		if( _outLayer != NULL )
			return _outLayer->count;
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
		_inLayer->cycle();

	}

	void backPushTargets( T t )
	{
		vecBackPrepTargets.push_back( t );
	}

	void backPropagate()
	{

		// * Calc error for layers
		_outLayer->calcError( vecBackPrepTargets );

		// * Calc gradients recursively
		_outLayer->calcGradient( vecBackPrepTargets );

		// Update weights recursively
		_outLayer->updateWeights( _learnRate, _momentum );

		//T outVal = _outLayer->nodes[0]->lastOut;

		vecBackPrepTargets.clear();

	}

	void store( std::string fileName )
	{
		XMLTag xml("NeuralNet");

		Layer<double>* layer = _inLayer;

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
			else if( act == tangenth )
			{
				activation = "tangenth";
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
	}

	void load( std::string fileName )
	{
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
			if( activation[0] == 'l' )
				pLayer = addLayer( count, linear, bias );
			else if( activation[0] == 's' )
				pLayer = addLayer( count, sigmoid, bias );
			else if( activation[0] == 't' )
				pLayer = addLayer( count, tangenth, bias );
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
	}
};

int main( int argc, char**argv)
{

	if( argc < 3 )
	{
		printf("\nusage: ann -w [(r/w)weights (restore) file name] [-i input file ] { -t training_file -x training_iterations -r learn_rate -m momentum -l [Layer spec] }\n");
		printf("\nexample: ./ann -w test.weights.xml -r 0.00002 -m 0.0002 -t train.txt -x 10 -i input.txt -l S2 S3 S2 S1\n\n");
		printf( "Layer types must be L, S, T, C, or e prefixed to the Node count.\n" );

		exit(1);
	}

	srand( time(NULL) );

	std::string strTrainingFile, strInputFile, strWeights ( "temp.weights.xml" );
	double lr=0.0, mo=0.0;
	int i = 1, training_iterations=1, times=1;
	FILE *t_fp = NULL, *i_fp = NULL;
	bool bias = false;
	bool cont = false;

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
			case 'i':
				++i;
				strInputFile = argv[i];
				++i;
				i_fp = fopen( strInputFile.c_str(), "r" );
				break;
			case 'b':
				++i;
				bias = true;
				break;
			case 'v':
				++i;
				g_verbose = 1;
				if( argv[i][0] != '-' )
				{
					g_verbose = atoi( argv[i] );
				}
				++i;
				break;
			case 'o':
				++i;
				g_output = 1;
				if( argv[i][0] != '-' )
				{
					g_output = atoi( argv[i] );
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
				t_fp = fopen( strTrainingFile.c_str(), "r" );
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
						switch( argv[i][0] )
						{
							case 'L':
								NN.addLayer( atoi( &argv[i][1] ), linear, bias );
								break;
							case 'S':
								NN.addLayer( atoi( &argv[i][1] ), sigmoid, bias );
								break;
							case 'T':
								NN.addLayer( atoi( &argv[i][1] ), tangenth, bias );
								break;
							case 'N':
								NN.addLayer( atoi( &argv[i][1] ), none, bias );
								break;
							case '-':
								break;
							default:
								printf( "Layer types must be L, S or T prefixed to the Node count.\n" );
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

		for( int e=0; e < training_iterations; e++ )
		{
			fseek( t_fp, 0, SEEK_SET );

			int ic = NN.getInputNodeCount();
			int oc = NN.getOutputNodeCount();
			char *line = NULL;
			size_t len = 0;
			ssize_t read;
			char *pch;

			char tmpline[1024];

			while( (read = getline(&line, &len, t_fp)) != -1 )
			{
				// Cycle inputs
				memcpy( tmpline, line, len+1 );

				for( int x = 0; x < times; x++ )
				{
					double val;
					pch = strtok (line," \t,:");
					for( int t=0; (t < ic) && (pch != NULL); t++ )
					{
						sscanf (pch, "%lf\n",&val);
						pch = strtok (NULL, " \t,:");
						NN.setInput( t, val );
						log_output( "I%d=%lf ", t, val );
					}
					NN.cycle();

					// Set targets for back propagation (training)
					for( int t=0; (t < oc) && (pch != NULL); t++ )
					{
						sscanf (pch, "%lf\n",&val);
						pch = strtok (NULL, " \t,:");
						NN.backPushTargets( val );
						log_output( "O%d=%lf ", t, val );
					}

					NN.backPropagate();

					log_output( "[%f]<%f>", NN.getOutput(0), val - NN.getOutput(0) );

					log_output( "\n" );

					g_counter++;
				}
			}
		}

		if( cont != true )
			NN.store( strWeights.c_str() );

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

		if( g_output > 0 ) g_output = 1;

		while( (read = getline(&line, &len, i_fp)) != -1 )
		{
			// Cycle inputs
			double val;
			pch = strtok (line," \t,:");
			for( int t=0; (t < ic) && (pch != NULL); t++ )
			{
				sscanf (pch, "%lf\n",&val);
				pch = strtok (NULL, " \t,:");
				NN.setInput( t, val );
				log_output( "i%d=%lf ", t, val );
			}
			NN.cycle();

			// Set targets for back propagation (training)
			for( int t=0; (t < oc); t++ )
			{
				log_output( "o%d=", t );
				printf( "%lf ", NN.getOutput(t) );
			}

			//printf( "[%f]", NN.getOutput(0) );

			printf( "\n" );
		}

	}

	return 0;
};
