

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

#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>


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

#define  PRINT_VEC_FORWARD( X )  if( ::_verbose != 0 ) { std::cout << "\n[" << __LINE__ << "](" << #X << ") -> "; for( T n : X )  std::cout << n << " : "; std::cout << "\n"; } 
//#define  PRINT_VEC_FORWARD( X )

#define  PRINT_VEC_BACK( X )     if( ::_verbose != 0 ) { std::cout << "\n{" << __LINE__ << "}(" << #X << ") <- "; for( T n : X )  std::cout << n << " : "; std::cout << "\n"; } 
//#define  PRINT_VEC_BACK( X )



enum ActType{ linear = 0, sigmoid, tangenth, relu, relul, softMax, none, bias };

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

	std::vector< thrust::host_vector<T> >       vecLayers;
	std::vector< thrust::host_vector<T> >       vecWeights;
	std::vector< thrust::host_vector<T> >       vecDeltas;
	std::vector< thrust::host_vector<T> >       vecGrads;
	std::vector< thrust::host_vector<int> >     vecForwardWeightKeys;
	std::vector< thrust::host_vector<int> >     vecBackwardWeightKeys;
	std::vector< thrust::host_vector<T> >       vecForwardSums;
	std::vector< thrust::host_vector<T> >       vecBackwardSums;

	std::vector<T> vecBackPrepTargets;

	NeuralNet( T learn_rate = 0.0001, T momentum = 0.001 )
		: _learnRate( learn_rate ), _momentum( momentum ), _dirty(false)
	{
	}

    ~NeuralNet()
    {
        clear();
    }

    //===================================================

    //template<typename T>
    static __host__ __device__ T actSigmoid( T n )
    {
	    return 1.0 / ( 1.0 + exp(-n) );
    }

    //template<typename T>
    static __host__ __device__ T derivSigmoid( T n )
    {
	    return n * ( (T)1.0 - n );
    }

    static __host__ __device__ T diff2DerivSigmoid(  T d, T o  )
    { 
        return 2.0*d*derivSigmoid(o); 
    }

    static __host__ __device__ T sDerivSigmoid(  T s, T o  )
    { 
        return s*derivSigmoid(o); 
    }

    static __host__ __device__ T diffSquared2( T t, T o )
    {
        T r = t - o;
	    return (r*r)/2.0;
    }

    static __host__ __device__ T addSquareOver2( T a, T b )
    {
	    return a+((b*b)/2.0);
    }

    //template<typename T>
    T actLinear( T n )
    {
	    return n;
    }

    //template<typename T>
    T derivLinear( T n )
    {
	    return 1.0;
    }



    //template<typename T>
    T actSoftMax( T n )
    {
	    return exp(n);
    }

    //template<typename T>
    T derivSoftMax( T n )
    {
	    return n * ( (T)1.0 - n );
    }

    //template<typename T>
    T actTanh( T n )
    {
	    return tanh( n );
    }

    //template<typename T>
    T derivTanh( T n )
    {
	    return 1.0 - n * n;
    }


    //template<typename T>
    T actReLUL( T n )
    {
	    return (n > 0.0) ? n : (n*.001);
    }

    //template<typename T>
    T derivReLUL( T n )
    {
	    return 1.0;
	    //return (n > 0.0) ? 1.0 : 0.0;
    }


    //===================================================

	T calcError( std::vector<T> &targets )
	{
		T netErr = (T)0.0, delta;

		return netErr;
	}

    T lastError()
    {
        return _lastError;
    }

	void cycle( int dropout_probability_select_mod = 0, T dropout_probability = 1.0 )
	{
		log_verbose("\n");

        int sz = vecLayers.size();

        thrust::equal_to<int> eq;
        thrust::plus<T> sum;
        thrust::multiplies<T> mul;

        for( int i=0; i < (sz-1); i++ )
        {

            thrust::host_vector<T> weighIn;

            auto &L1 = vecLayers[i];
            auto &L2 = vecLayers[i+1];

            int L1_sz = L1.size();
            int L2_sz = L2.size();

            //auto layer = L1;

    PRINT_VEC_FORWARD( L1 )

            ActType act = sigmoid;


            switch( act )
            {
		        case linear:
                    break;
		        case sigmoid:
                    // Run activation function
                    thrust::transform( L1.begin(), L1.end(), L1.begin(), actSigmoid );
                    break;
		        case tangenth:
                    break;
		        case softMax:
                    break;
		        case relu:
		            break;
		        case relul:
                    break;	            
                default:
                    return;
            }

    PRINT_VEC_FORWARD( L1 )

            for( T n : L1 )
                for( int d=0; d < L2_sz; d++ ) 
                    weighIn.push_back( n );
    
    PRINT_VEC_FORWARD( weighIn )
    
            thrust::host_vector<T> weighOut( L2_sz ); 
            thrust::host_vector<int> weighOutKeys( L2_sz ); 

            // Create a holding vector for repeated layer activation results
            thrust::host_vector<T> tmp_vec;

            // Copy results into tmp_vec repeatedly
            for( int r=0; r < L1_sz; r++ )
                for( int nl=0; nl < L2_sz; nl++ )
                    tmp_vec.push_back( L1[r] );

    PRINT_VEC_FORWARD( tmp_vec ) 
    PRINT_VEC_FORWARD( vecWeights[i] ) 

            thrust::transform( tmp_vec.begin(), tmp_vec.end(), vecWeights[i].begin(), tmp_vec.begin(), mul );

    PRINT_VEC_FORWARD( vecForwardWeightKeys[i] )
    PRINT_VEC_FORWARD( tmp_vec ) 
    PRINT_VEC_FORWARD( weighOutKeys )
    PRINT_VEC_FORWARD( weighOut ) 

            thrust::reduce_by_key( thrust::host, 
                        vecForwardWeightKeys[i].begin(), 
                        vecForwardWeightKeys[i].end(),
                        tmp_vec.begin(),
                        weighOutKeys.begin(),
                        weighOut.begin(),
                        eq, sum );
                        
    PRINT_VEC_FORWARD( weighOutKeys )
    PRINT_VEC_FORWARD( weighOut )
    PRINT_VEC_FORWARD( L2 ) 

            thrust::copy( weighOut.begin(), weighOut.end(), L2.begin() );
 
    PRINT_VEC_FORWARD( L2 )
        }        

    PRINT_VEC_FORWARD( vecLayers.back() )

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

        // Add weights after input layer
        if( !vecLayers.empty() )
        {
            vecWeights.push_back( thrust::host_vector<T>( n*vecLayers.back().size() ) );
            auto &ws = vecWeights.back();
            thrust::generate( ws.begin(), ws.end(), [](){ return (T)std::rand() / (T)RAND_MAX; } );

            vecDeltas.push_back( thrust::host_vector<T>( n*vecLayers.back().size() ) );
            thrust::fill( vecDeltas.back().begin(), vecDeltas.back().end(), 0 );

            vecForwardWeightKeys.push_back( thrust::host_vector<T>(  ) );

            auto &fwk = vecForwardWeightKeys.back();
            for( int j=0; j<n; j++ )
                for( int i=0; i<vecLayers.back().size(); i++ )
                    fwk.push_back( j );
                
            vecBackwardWeightKeys.push_back( thrust::host_vector<T>(  ) );

            auto &bwk = vecBackwardWeightKeys.back();
            for( int i=0; i<vecLayers.back().size(); i++ )
                for( int j=0; j<n; j++ )
                    bwk.push_back( j );
                
        }

		vecLayers.push_back( thrust::host_vector<T>(n) );

        vecGrads.push_back( thrust::host_vector<T>( n ) );
        thrust::fill( vecGrads.back().begin(), vecGrads.back().end(), 0 );


        return n;
	}


	void calcGradient( std::vector<T> &targets )
	{   
        _lastError = (T)0.0;

        auto &L = vecLayers.back();

        thrust::host_vector<T> diffs( L.size() );
     
        PRINT_VEC_BACK( L );
        PRINT_VEC_BACK( targets );

        thrust::transform( L.begin(), L.end(), targets.begin(), diffs.begin(), [](T a, T b) -> T { return a-b; } ); // Diffs

        PRINT_VEC_BACK( L );
        PRINT_VEC_BACK( targets );
        PRINT_VEC_BACK( diffs );

        //_lastError = thrust::reduce( diffs.begin(), diffs.end(), 0, [](T a, T b) -> T { return a+((b*b)/2.0); } );

        for( auto &d : diffs )
            _lastError += (d*d)/2.0;

        //std::cout << _lastError << std::endl;
        _lastError /= (T)L.size();
        //std::cout << _lastError << std::endl;
        _lastError = sqrt( _lastError );
        //std::cout << _lastError << std::endl;

        
        //_lastError = thrust::inner_product( L.begin(), L.end(), targets.begin(), 0, thrust::plus<T>(), diffSquared2<T> );


        int sz = vecLayers.size();
        for( int i=sz; i > 1; i-- )
        {
            auto &W = vecWeights[i-1];
            auto &Wk = vecForwardWeightKeys[i-1];
            auto &L = vecLayers[i-1];
            auto &L2 = vecLayers[i-2];
            auto &G = vecGrads[i-1];
            thrust::host_vector<T> sumOut( L2.size() ); 
            thrust::host_vector<int> sumOutKeys( L2.size() ); 

            if( i == sz ) // output layer
            {
                //std::cout << i << " : " << vecGrads.size() << std::endl;
                PRINT_VEC_BACK( L )
                PRINT_VEC_BACK( diffs )
                PRINT_VEC_BACK( G )
                //thrust::transform( L.begin(), L.end(), diffs.begin(), vecGrads[i-1].begin(), [](T o,T d) -> T { return 2.0*d*derivSigmoid(o); } );
                thrust::transform( L.begin(), L.end(), diffs.begin(), G.begin(), diff2DerivSigmoid );
                PRINT_VEC_BACK( G )
            }
            else
            {
                auto &G2 = vecGrads[i];
                PRINT_VEC_BACK( Wk )
                PRINT_VEC_BACK( W )
                PRINT_VEC_BACK( L )
                PRINT_VEC_BACK( L2 )
                PRINT_VEC_BACK( G2 )
                PRINT_VEC_BACK( G )


                int kmx = Wk.back()+1;  // Get assumed max key
                int ksz = Wk.size();
                int kmod= ksz / kmx;

                for( int g=0; g < G.size(); g++ )
                {
                    T sum = 0.0;
                    for( int x=0; x < kmod; x++ )
                    {
                        //int y = x+(g*kmod);
                        int y = g+(x*kmod);
                        sum += G2[x] * W[y];
                    }
                    G[g] = sum * derivSigmoid( L[g] );
                }

                PRINT_VEC_BACK( G )

                thrust::host_vector<T> tmp_vec;
                thrust::host_vector<T> tmp_vec_sum( L.size() );
                thrust::host_vector<T> tmp_vec_grad_sum( G.size() );

                //thrust::transform( G.begin(), G.end(), W.begin(), G.begin(), [](T w,T g){ return (w*g); } );

/*
                thrust::reduce_by_key( thrust::host, 
                        //vecBackwardWeightKeys[i-1].begin(), 
                        //vecBackwardWeightKeys[i-1].end(),
                        vecForwardWeightKeys[i-1].begin(), 
                        vecForwardWeightKeys[i-1].end(),
                        tmp_vec.begin(),
                        sumOutKeys.begin(),
                        sumOut.begin(),
                        thrust::equal_to<T>(), thrust::plus<T>() );
*/
                PRINT_VEC_BACK( vecForwardWeightKeys[i-1] )
                PRINT_VEC_BACK( vecBackwardWeightKeys[i-1] )
                PRINT_VEC_BACK( G2 )
                PRINT_VEC_BACK( sumOutKeys )
                PRINT_VEC_BACK( sumOut )



                PRINT_VEC_BACK( L2 )
                PRINT_VEC_BACK( sumOut )

                //thrust::transform( sumOut.begin(), sumOut.end(), L2.begin(), sumOut.begin(), [](T s,T o) -> T { return (s*derivSigmoid(o)); } );
                //thrust::transform( sumOut.begin(), sumOut.end(), L2.begin(), sumOut.begin(), sDerivSigmoid );

                PRINT_VEC_BACK( L2 )
                PRINT_VEC_BACK( sumOut )

            }
        }


        
	}


	void updateWeights( T learnRate, T momentum )
	{

        T delta, grad, out, weight;

        auto &L = vecLayers.back();


        int sz = vecLayers.size();
        for( int i=sz-2; i > 1; i-- )
        {
            auto &W = vecWeights[i];
            auto &Wk = vecForwardWeightKeys[i];
            auto &L = vecLayers[i+1];
            auto &D = vecDeltas[i];
            auto &L2 = vecLayers[i];
            auto &G = vecGrads[i];

            thrust::host_vector<T> sumOut( L2.size() ); 
            thrust::host_vector<int> sumOutKeys( L2.size() ); 

        
            auto &G2 = vecGrads[i+1];
            PRINT_VEC_BACK( Wk )
            PRINT_VEC_BACK( W )
            PRINT_VEC_BACK( L )
            PRINT_VEC_BACK( L2 )
            PRINT_VEC_BACK( G )
            PRINT_VEC_BACK( G2 )


            int kmx = Wk.back()+1;  // Get assumed max key
            int ksz = Wk.size();
            int kmod= ksz / kmx;

            for( int n=L.size()-1; n >= 0 ; n-- )
            {
                for( int x=0; x < kmod; x++ )
                {
                    int y = x+(n*kmod);
                    delta = D[y];
                    grad = G[n];
                    out = L[n];
                    weight = W[y];

                    delta = (learnRate * grad * out) + (momentum * delta);

                    D[y] = delta;
                    W[y] += delta;
                }
            }
        }
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

		// * Calc gradients recursively
		calcGradient( vecBackPrepTargets );

		// Update weights recursively
		updateWeights( _learnRate, _momentum );

		//T outVal = _outLayer->nodes[0]->lastOut;

		vecBackPrepTargets.clear();

        return _lastError;
	}

	void store( std::string fileName )
	{

        if( !_dirty )
            return;

        std::ofstream myfile (fileName);

        if (myfile.is_open())
        for( auto &L : vecLayers )
        {
            myfile << "sigmoid:" << L.size();
            for( auto &N : L )
            {
                myfile << ":";
                myfile << N;
            }
            myfile << "\n";
        }
        for( auto &Ws : vecWeights )
        {
            myfile << "weights:" << Ws.size();
            for( auto &w : Ws )
            {
                myfile << ":";
                myfile << w;
            }
            myfile << "\n";
        }
        myfile.close();

        _dirty = false;
	}

	void load( std::string fileName )
	{

		ssize_t read;
        int iNodeCount;
        size_t len;
		char *pch;
        char *line = NULL;

        FILE* fp = fopen( fileName.c_str(), "rb" );

        if( fp != NULL )
            line = (char*)malloc( 16*1024 );


        int w=0;
        if( fp != NULL )
	    while( (read = getline( &line, &len, fp) ) != -1 )
	    {
            char szType[256];

	        dataType val;
	        pch = strtok (line," \t,:");
            //std::cout << "Reading Line: " << line << std::endl;

            if( strncmp( "weights", line, 7 ) != 0 ) // Layers
            {
	            for( int t=0; (pch != NULL); t++ )
	            {
                    if( t == 0 )
                    {
                        sscanf (pch, "%s\n",(char*)&szType);
		                pch = strtok (NULL, " \t,:");
                    }
                    else if( t == 1 )
                    {
                        sscanf (pch, "%d\n",&iNodeCount);
		                pch = strtok (NULL, " \t,:");
                        addLayer( iNodeCount, sigmoid, false );
                        //std::cout << "Adding " << szType << " layer with " << iNodeCount << " nodes\n";
                    }  
                    else
                    {
		                sscanf (pch, "%lf\n", &val);
		                pch = strtok (NULL, " \t,:");
                        vecLayers.back()[t-2] = val;
                        //std::cout << val << "  ";      
                    }

                    //std::cout << std::endl;
       	        }
            }
            else // Weights
            {
	            for( int t=0; (pch != NULL); t++ )
	            {
                    if( t == 0 )
                    {
                        sscanf (pch, "%s\n",(char*)&szType);
		                pch = strtok (NULL, " \t,:");
                    }
                    else if( t == 1 )
                    {
                        sscanf (pch, "%d\n",&iNodeCount);
		                pch = strtok (NULL, " \t,:");
                        //addLayer( iNodeCount, sigmoid, false );
                        //std::cout << "Adding weights with " << iNodeCount << " weights\n";
                    }  
                    else
                    {
		                sscanf (pch, "%lf\n", &val);
		                pch = strtok (NULL, " \t,:");
                        vecWeights[w][t-2] = val;
                        //std::cout << val << "  ";      
                    }

                    //std::cout << std::endl;
       	        }
                w++;
            }
        }

        fclose( fp );




        _dirty = false;
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

			//printf( "\n" );
		}

	}
                


	return 0;
};
