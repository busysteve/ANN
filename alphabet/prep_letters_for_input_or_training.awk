{
    

    if( length($0) != 0 && $1 != "=" )
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
    else if( $1 == "=" )
    {
        switch($2)
        {
            case "A": out = "0 0 0 0 1"; break;
            case "B": out = "0 0 0 1 0"; break;
            case "C": out = "0 0 0 1 1"; break;
            case "D": out = "0 0 1 0 0"; break;
            case "E": out = "0 0 1 0 1"; break;
            case "F": out = "0 0 1 1 0"; break;
            case "G": out = "0 0 1 1 1"; break;
            case "H": out = "0 1 0 0 0"; break;
            case "I": out = "0 1 0 0 1"; break;
            case "J": out = "0 1 0 1 0"; break;
            case "K": out = "0 1 0 1 1"; break;
            case "L": out = "0 1 1 0 0"; break;
            case "M": out = "0 1 1 0 1"; break;
            case "N": out = "0 1 1 1 0"; break;
            case "O": out = "0 1 1 1 1"; break;
            case "P": out = "1 0 0 0 0"; break;
            case "Q": out = "1 0 0 0 1"; break;
            case "R": out = "1 0 0 1 0"; break;
            case "S": out = "1 0 0 1 1"; break;
            case "T": out = "1 0 1 0 0"; break;
            case "U": out = "1 0 1 0 1"; break;
            case "V": out = "1 0 1 1 0"; break;
            case "W": out = "1 0 1 1 1"; break;
            case "X": out = "1 1 0 0 0"; break;
            case "Y": out = "1 1 0 0 1"; break;
            case "Z": out = "1 1 0 1 0"; break;
            default: out = "0 0 0 0 0"; break;
        }
    }
    else
    {
        print letter_input "   " out;
        letter_input = "";

        #printf("\n");
    }
    
}
