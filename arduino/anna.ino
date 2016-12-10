
enum ActType{ 
  linear = 0, sigmoid, tangenth, none, bias };


template <class T>
class funcs
{
  public:
  static T actBias( T n )
  {
    return 1.0;
  }

  static T actNone( T n )
  {
    return n;
  }

  static T actLinear( T n )
  {
    return n;
  }

  static T actSigmoid( T n )
  {
    return 1.0 / ( 1.0 + exp(-n) );
  }
    
  static T derivLinear( T n )
  {
    return 1.0;
  }
  
  static T derivSigmoid( T n )
  {
    return n * ( 1.0 - n );
  }
};


// Minimal class to replace std::vector
template<class Data>
class Vector {
  size_t d_size; // Stores no. of actually stored objects
  size_t d_capacity; // Stores allocated capacity
  Data *d_data; // Stores data
public:
  Vector() : 
  d_size(0), d_capacity(0), d_data(0) {
  }; // Default constructor
  Vector(Vector const &other) : 
  d_size(other.d_size), d_capacity(other.d_capacity), d_data(0) { 
    d_data = (Data *)malloc(d_capacity*sizeof(Data)); 
    memcpy(d_data, other.d_data, d_size*sizeof(Data)); 
  }; // Copy constuctor
  ~Vector() { 
    free(d_data); 
  }; // Destructor
  Vector &operator=(Vector const &other) { 
    free(d_data); 
    d_size = other.d_size; 
    d_capacity = other.d_capacity; 
    d_data = (Data *)malloc(d_capacity*sizeof(Data)); 
    memcpy(d_data, other.d_data, d_size*sizeof(Data)); 
    return *this; 
  }; // Needed for memory management
  void push_back(Data const &x) { 
    if (d_capacity == d_size) resize(); 
    d_data[d_size++] = x; 
  }; // Adds new value. If needed, allocates more space
  size_t size() const { 
    return d_size; 
  }; // Size getter
  Data const &operator[](size_t idx) const { 
    return d_data[idx]; 
  }; // Const getter
  Data &operator[](size_t idx) { 
    return d_data[idx]; 
  }; // Changeable getter
private:
  void resize() { 
    d_capacity = d_capacity ? d_capacity*2 : 1; 
    Data *newdata = (Data *)malloc(d_capacity*sizeof(Data)); 
    memcpy(newdata, d_data, d_size * sizeof(Data)); 
    free(d_data); 
    d_data = newdata; 
  };// Allocates double the old space
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

  Vector<Connection*> conns, inConns;

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

  Vector<Node<T>*> nodes;

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

  Layer( int n, ActType act, bool bias )
: 
    count(n), prevLayer(NULL), nextLayer(NULL), _activation(act), _bias(bias),
    sumIn(0.0)
    {

      if( act == linear )
      {
        _actFunc = funcs<T>::actLinear;
        _derivActFunc = funcs<T>::derivLinear;
      }
      else if( act == sigmoid )
      {
        _actFunc = funcs<T>::actSigmoid;
        _derivActFunc = funcs<T>::derivSigmoid;
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
        _actFunc = funcs<T>::actNone;
        _derivActFunc = funcs<T>::actNone;
      }

      for( int i=0; i < count; i++ )
      {
        nodes.push_back( new Node<T>( _actFunc, false ) );
      }

      if( bias == true )
      {
        nodes.push_back( new Node<T>( funcs<T>::actBias, true ) );
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

  T calcError( Vector<T> &targets )
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

  void calcGradient( Vector<T> &targets )
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

  Vector<Layer<T>*> layers;

  Vector<T> vecBackPrepTargets;

public:

  NeuralNet( T learn_rate = 0.01, T momentum = 0.0001 )
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


NeuralNet<double> nn( 0.05, 0.0001 );

void setup()
{
  nn.addLayer( 2, linear, bias );
  nn.addLayer( 2, sigmoid, bias );
  nn.addLayer( 1, linear, bias );

}


void loop()
{


}


