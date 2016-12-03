{
        
    if( length($0) != 0 )
    {
        for( i=1; i <= length($0); i++ )
        {
            if( substr( $0, i, 1 ) == " " )
                letter_input = letter_input "0" " ";
            else
                letter_input = letter_input "1" " ";
        }
    }
    else
    {
        counter += 1
        print letter_input " " counter / 100;
        letter_input = "";

        #printf("\n");
    }
    
}
