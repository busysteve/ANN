#include <stdio.h>
#include <stdlib.h>

int main( int argc, char **argv )
{
	int offset= atoi(argv[1]);
	int count = atoi(argv[2]);
	int min   = atoi(argv[3]);
	int max   = atoi(argv[4]);
	char* on_c  = argv[5];
	char* off_c = argv[6];
	char* datafile  = argv[7];
	char* labelfile = argv[8];

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
				if( stash[i][j] > min && stash[i][j] < max )
					printf("%s", on_c);
				else
					printf("%s", off_c);
			}
			printf("\n");
		}

		fread( &stash[0][0], 1, 1, labels );
		printf( "%d\n", stash[0][0] );
	}

	return 0;
}



