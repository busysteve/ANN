{
    

    if( length($0) != 0 )
    {
        line = $0;

        for( i=length(line); i < 7; i++ )
            line = line " ";

        for( i=1; i <= length(line); i++ )
        {
            if( substr( line, i, 1 ) == " " )
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
