{
	if( length($0) == 1 )
		switch( $1 )
		{
			case "0":
				printf("1 0 0 0 0 0 0 0 0 0\n");
				break;
			case "1":
				printf("0 1 0 0 0 0 0 0 0 0\n");
				break;
			case "2":
				printf("0 0 1 0 0 0 0 0 0 0\n");
				break;
			case "3":
				printf("0 0 0 1 0 0 0 0 0 0\n");
				break;
			case "4":
				printf("0 0 0 0 1 0 0 0 0 0\n");
				break;
			case "5":
				printf("0 0 0 0 0 1 0 0 0 0\n");
				break;
			case "6":
				printf("0 0 0 0 0 0 1 0 0 0\n");
				break;
			case "7":
				printf("0 0 0 0 0 0 0 1 0 0\n");
				break;
			case "8":
				printf("0 0 0 0 0 0 0 0 1 0\n");
				break;
			case "9":
				printf("0 0 0 0 0 0 0 0 0 1\n");
				break;
		}
	else
		printf( "%s", $0 );	
}
