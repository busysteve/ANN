
enum ActType{ 
  linear = 0, sigmoid, tangenth, none, bias };

template <class T>
T actBias( T n )
{
  return (T)1.0;
}

template <typename T>
T actNone( T n )
{
  return n;
}

template <typename T>
T actLinear( T n )
{
  return n;
}


template <typename T>
T actSigmoid( T n )
{
  return 1.0 / ( 1.0 + exp(-n) );
}


template <typename T>
T derivLinear( T n )
{
  return 1.0;
}


template <typename T>
T derivSigmoid( T n )
{
  return n * ( 1.0 - n );
}



template<class T>
class vector
{
  T _vec;
  int _sz;
  int _trk;

public:
  vector(int sz=10) : 
  _sz(sz), _trk(0) { 
    _vec = malloc( sizeof(T) * sz ); 
  };
  ~vector(){ 
    free(_vec); 
  };
  void push_back( T p )
  { 
    ((T)(_vec+_trk)) = p; 
    _trk++; 
    if( _trk > _sz ) 
    { 
      _sz+=10;
      _vec = realloc( _vec, sizeof(T) * _sz ); 
    }
  };
  T operator[](int i){ 
    return _vec+i; 
  };

  boolean empty() { 
    return _trk == 0; 
  };
  int size() { 
    return _trk; 
  };


};

template<typename T>
class Node
{
public:

  T inSum, lastOut;
  T deltaErr;
  T grad;
  bool _bias;

  class Connection;

  vector<Connection*> conns, inConns;

  bool _activate;

  //ActType _activation;

  typedef T ( *ActFunc )(T);

  ActFunc _actFunc;

  Node( ActFunc actFunc, boolean bias ) :
  inSum((T)0.0), lastOut((T)0.0),
  deltaErr((T)0.0), grad((T)1.0),
  _actFunc(actFunc), _bias(bias), _activate(false)
  {

  }

  void input( T in )
  {
    inSum += in;			 // Sum weighted inputs for activation
  }

  T out()
  {
    return ( lastOut );
  }

  // Node to bind to (next layer node)
  void bindNode( Node* node )
  {
    Connection* pConn = new Connection( this, node );
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

    inSum = (T)0.0;

    return lastOut;
  }

  //template<typename T>
  class Connection
  {
public:

    T weight, alpha, delta;

    Node *fromNode, *toNode;


    Connection( Node* fNode, Node* tNode )
: 
      fromNode( fNode ), toNode( tNode ), alpha((T)1.0), delta((T)0.0)
      {
        weight = random(1, 1000000) * 0.000001;
      }

    void xmit( T in )
    {
      if( toNode != NULL )
      {
        // Apply weight here
        toNode->in( in * weight );
      }
    }

  };

  void in( T in )
  {
    _activate = true;
    inSum += in;			 // Sum weighted inputs for activation
  }


};


template<typename T>
class Layer
{
public:

  vector<Node<T>*> nodes;

  Layer<T>* prevLayer;
  Layer<T>* nextLayer;

  ActType _activation;

  typedef T ( *derivActFunc )(T);
  typedef T ( *actFunc )(T);

  derivActFunc _derivActFunc;
  actFunc _actFunc;

  bool _bias;

  int count;

  T sumIn;

  String _name;

  Layer( int n, int act, bool bias )
: 
    count(n), prevLayer(NULL), nextLayer(NULL), _activation(act), _bias(bias),
    sumIn(0.0)
    {

      if( act == linear )
      {
        _actFunc = actLinear;
        _derivActFunc = derivLinear;
      }
      else if( act == sigmoid )
      {
        _actFunc = actSigmoid;
        _derivActFunc = derivSigmoid;
      }
      /*
      else if( act == tangenth )
       {
       _actFunc = actTanh;
       _derivActFunc = derivTanh;
       }
       */
      else if( act == none )
      {
        _actFunc = actNone;
        _derivActFunc = actNone;
      }

      for( int i=0; i < count; i++ )
      {
        nodes.push_back( new Node<T>( _actFunc, false ) );
      }

      if( bias == true )
      {
        nodes.push_back( new Node<T>( actBias, true ) );
      }
    }

  void bindLayer( Layer* layer )
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

  T calcError( vector<T> &targets )
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

  T sumDOW( Layer *nLayer )
  {
    T sum = 0.0;
    int ns = nLayer->nodes.size()-( nLayer->_bias ? 1 : 0 );

    for( int n = 0; n < ns; n++ )
    {
      for( int c = 0; c < nLayer->nodes[n]->conns.size(); c++ )
      {
        T grad = nLayer->nodes[n]->conns[c]->toNode->grad;
        sum += ( nLayer->nodes[n]->conns[c]->weight ) * grad;

      }
    }

    return sum;
  }

  void calcGradient( vector<T> &targets )
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
        }

        nodes[n]->grad = sum * _derivActFunc( nodes[n]->lastOut );

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
          typename Node<T>::Connection* conn = nodes[i]->inConns[c];
          delta = conn->delta;
          grad = nodes[i]->grad;
          //grad = conn->fromNode->grad;
          //out = nodes[i]->lastOut;
          out = conn->fromNode->lastOut;
          weight = conn->weight;

          delta = learnRate * grad * out + momentum * delta;

          conn->delta = delta;
          conn->weight += delta;
        }
      }

      prevLayer->updateWeights( learnRate, momentum );
    }
  }

  void cycle(  )
  {

    sumIn = 0.0;

    for( int i=nodes.size()-1; i>=0; i-- )
      sumIn += nodes[i]->cycle();

    if( nextLayer != NULL )
      nextLayer->cycle( );

  }

};

template< class T >
class NeuralNet
{

  T _learnRate;
  T _momentum;
  Layer<T> *_inLayer, *_outLayer;

  vector<Layer<T>*> layers;

  vector<T> vecBackPrepTargets;

  NeuralNet( T learn_rate = 0.0001, T momentum = 0.001 )
: 
    _learnRate( learn_rate ), _momentum( momentum )
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

    layers.push_back( pl = new Layer<T>(n, act, bias ) );

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

};



