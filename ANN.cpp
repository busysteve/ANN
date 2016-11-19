
// g++ -g -o ann ANN.cpp

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

const void* nullptr = NULL;

template<typename T>
struct Node;


template<typename T>
struct Connection
{

    T weight, deltaWeight;

    Node<T> *toNode;

	//Connection( void Node<T>::(*iofunc)( T ) ) : output(iofunc) 
	Connection( Node<T>* node ) : toNode( node ) 
    {
        T rnd = (T)std::rand() / RAND_MAX;

        weight = ( rnd * (T)0.5 ) - ( (T)0.5 / (T)2.0 );
    }

	//void Node<T>::(*output)( T );

	void xmit( T in )
	{
		if( toNode != nullptr )
        {
			toNode->input( in*weight ); // Apply weight here
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

    Node() : inSum((T)0.0), lastOut((T)0.0), deltaErr((T)0.0), grad((T)0.0) {}

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
            
            lastOut = out;

            inSum = (T)0.0;
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
        for( int i=0; i < n; i++ )
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

    void activate()
    {
        for( int i=0; i<count; i++ )
        {
            nodes[i]->activate();
        }

        if( nextLayer != NULL )
            nextLayer->activate();
    }

};




int main()
{


    double learnRate = 0.05;
    double momentum = 0.05;

    Layer<double>   inLayer(1);
    Layer<double>   hiddenLayer1(2);
    Layer<double>   hiddenLayer2(2);
    //Layer<double>   hiddenLayer3(2);
    Layer<double>   outLayer(1);

    inLayer.bindLayer( &hiddenLayer1 );
    hiddenLayer1.bindLayer( &hiddenLayer2 );
    //hiddenLayer2.bindLayer( &hiddenLayer3 );
    hiddenLayer2.bindLayer( &outLayer );

    double o = 0.0;
    double outErr;

    for( double t=1.0, d=0.0, e=0.0; t <= 100; t++ )
    {
        //double target = t*t;
        //double target = t*t;
        double target = sin(t);
        int i;
        inLayer.nodes[0]->input(target);

        inLayer.activate();

        o = outLayer.nodes[0]->lastOut;



        //e = d = target - o;  // Calc error
        // * Calc gradient for hidden layers
        Layer<double>* layer = outLayer.prevLayer;
        while( layer->prevLayer != NULL )
        {
            for( i=layer->nodes.size()-1; i>=0; i-- )
            {
                outErr = 0.0;
                for( int c = layer->nodes[i]->conns.size()-1; c >= 0; c-- ) 
                {
                    for( int nn = layer->nextLayer->nodes.size()-1; nn >=0; nn-- )
                        outErr +=  ( layer->nodes[i]->lastOut - target );
                                    //* ( layer->nodes[i]->lastOut - target );
                }
                layer->nodes[i]->grad = outErr * 1.0;
                layer->nodes[i]->deltaErr = outErr;
            }

            layer = layer->prevLayer;        

        }
        

        // * Calc gradient for output layer
        for( i=outLayer.nodes.size()-1; i>=0; i-- )
        {
            outErr = outLayer.nodes[i]->lastOut - target;
            outLayer.nodes[i]->grad = outErr * 1.0;
        }


        // * Calc gradient for hidden layers
        layer = outLayer.prevLayer;
        while( layer->prevLayer != NULL )
        {
            for( i=layer->nodes.size()-1; i>=0; i-- )
            {
                outErr = 0.0;
                for( int c = layer->nodes[i]->conns.size()-1; c >= 0; c-- ) 
                {
                    for( int nn = layer->nextLayer->nodes.size()-1; nn >=0; nn-- )
                        outErr += layer->nodes[i]->conns[c]->weight * layer->nextLayer->nodes[nn]->grad;
                }
                layer->nodes[i]->grad = outErr * 1.0;
            }

            layer = layer->prevLayer;        

        }




        // Update weights
        double oldDeltaWeight, newDeltaWeight;
        layer = outLayer.prevLayer;
        while( layer->prevLayer != NULL )
        {
            for( i=layer->nodes.size()-1; i>=0; i-- )
            for( int pn=layer->prevLayer->nodes.size()-1; i>=0; i-- )
            {
                outErr = 0.0;
                for( int c = layer->nodes[i]->conns.size()-1; c >= 0; c-- ) 
                {
                    oldDeltaWeight = layer->nodes[i]->conns[c]->deltaWeight;

                    newDeltaWeight =    learnRate 
                                        * layer->nodes[i]->grad
                                        / layer->prevLayer->nodes[pn]->out()
                                        + momentum
                                        * oldDeltaWeight;

                    layer->nodes[i]->conns[c]->deltaWeight = newDeltaWeight;
                    layer->nodes[i]->conns[c]->weight += newDeltaWeight;
                }
            }

            layer = layer->prevLayer;        

        }


        std::cout << target << "\t" << o << "\t" << target-o << std::endl;

    }

    return 0;
};




