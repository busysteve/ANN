{
	
	binary = "";

	for( i=1; i<=NF; i++ )
	{
		if( i > 1 ) binary = binary " ";
		binary = binary ($i > .5 ? "1":"0");
	}

	switch( binary )
	{
		case "0 0 0 0 1": out = "A"; break;
		case "0 0 0 1 0": out = "B"; break;
		case "0 0 0 1 1": out = "C"; break;
		case "0 0 1 0 0": out = "D"; break;
		case "0 0 1 0 1": out = "E"; break;
		case "0 0 1 1 0": out = "F"; break;
		case "0 0 1 1 1": out = "G"; break;
		case "0 1 0 0 0": out = "H"; break;
		case "0 1 0 0 1": out = "I"; break;
		case "0 1 0 1 0": out = "J"; break;
		case "0 1 0 1 1": out = "K"; break;
		case "0 1 1 0 0": out = "L"; break;
		case "0 1 1 0 1": out = "M"; break;
		case "0 1 1 1 0": out = "N"; break;
		case "0 1 1 1 1": out = "O"; break;
		case "1 0 0 0 0": out = "P"; break;
		case "1 0 0 0 1": out = "Q"; break;
		case "1 0 0 1 0": out = "R"; break;
		case "1 0 0 1 1": out = "S"; break;
		case "1 0 1 0 0": out = "T"; break;
		case "1 0 1 0 1": out = "U"; break;
		case "1 0 1 1 0": out = "V"; break;
		case "1 0 1 1 1": out = "W"; break;
		case "1 1 0 0 0": out = "X"; break;
		case "1 1 0 0 1": out = "Y"; break;
		case "1 1 0 1 0": out = "Z"; break;
		default:
			out = "?";
	}

	print out;

}
