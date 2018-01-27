#include <stdio.h>
#include <stdlib.h>

int main( int argc, char **argv )
{
	int offset= atoi(argv[1]);
	int count = atoi(argv[2]);
	int min   = atoi(argv[3]);
	int max   = atoi(argv[4]);
    float mul = atof(argv[5]);
	char* on_c  = argv[6];
	char* off_c = argv[7];
	char* datafile  = argv[8];
	char* labelfile = argv[9];
	int newline = atoi(argv[10]);

	unsigned char stash[28][28];

	FILE* data = fopen( datafile, "r" );
	FILE* labels=fopen( labelfile, "r");

	fread( stash, 1, 128/8, data );
	fread( stash, 1, 64/8, labels );

	for( int o=0; o < offset; o++ )
	{
		fread( stash, 28, 28, data );
		fread( stash, 1, 1, labels );
	}
		
	for( int c=0; c < count; c++ )
	{
		fread( stash, 28, 28, data );

		for( int i=0; i < 28; i++ )
		{
			for( int j=0; j < 28; j++ )
			{
                if( mul > 0.0 )
                {
				    printf("%1.3f ", ((float)stash[i][j] * mul) );
                }
                else
                {
				    if( stash[i][j] > min && stash[i][j] < max )
					    printf("%s", on_c);
				    else
					    printf("%s", off_c);
                }
			}
			if( newline > 0 ) printf("\n");
			//else printf(" ");
		}

		fread( &stash[0][0], 1, 1, labels );

		switch( stash[0][0] )
		{
			case 0:
				printf("1 0 0 0 0 0 0 0 0 0 ");
				break;
			case 1:
				printf("0 1 0 0 0 0 0 0 0 0 ");
				break;
			case 2:
				printf("0 0 1 0 0 0 0 0 0 0 ");
				break;
			case 3:
				printf("0 0 0 1 0 0 0 0 0 0 ");
				break;
			case 4:
				printf("0 0 0 0 1 0 0 0 0 0 ");
				break;
			case 5:
				printf("0 0 0 0 0 1 0 0 0 0 ");
				break;
			case 6:
				printf("0 0 0 0 0 0 1 0 0 0 ");
				break;
			case 7:
				printf("0 0 0 0 0 0 0 1 0 0 ");
				break;
			case 8:
				printf("0 0 0 0 0 0 0 0 1 0 ");
				break;
			case 9:
				printf("0 0 0 0 0 0 0 0 0 1 ");
				break;
		}

		printf( "%d\n", stash[0][0] );
	}

	return 0;
}



