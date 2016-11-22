{
    if( length($0) != 0 )
    {
        printf( "%s", $0 ); 

        for( i=length($0); i < 8; i++ )
            printf(" ");
    }

    printf("\n");
    
}
