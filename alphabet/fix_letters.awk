{
    if( length($0) != 0 )
    {
        printf( "%s", $0 ); 

        for( i=length($0); i < 7; i++ )
            printf(" ");
    }

    printf("\n");
    
}
