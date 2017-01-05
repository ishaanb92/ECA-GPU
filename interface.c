#include "interface.h"


bool requestInput(char* hash){
    printf("R\n");
    fflush(stdout);
    int cnt = scanf("%s", hash);
    FILE *check_base;
    check_base = fopen("./base.txt","a+");
    fprintf(check_base,"Base string : %s \n",hash);
    fclose(check_base);

    //check if command is accepted
    if (cnt==0 || strcmp(hash, "NAK")==0){
        hash[0]='\0';
        return false;
    }

    //duplicate input to get large input string
    for(int j=0;j<INPUT_MULT;j++)
        for(int i=0;i<UNIQUE_INPUT_SIZE;i++)
            hash[j*UNIQUE_INPUT_SIZE+i]=hash[i];
    hash[INPUT_SIZE]='\0';//terminate input string

    return true;
}

bool validateHash(char* hash, char* nonce){
    
    //send verify command
    hash[UNIQUE_INPUT_SIZE]='\0';
    if (nonce[0]!='\0')
        printf("V %s %s\n",hash, nonce);
    else
        printf("V %s NONE\n",hash); //send no match was found
    fflush(stdout);
    
    //check response
    char response[10];
    int cnt = scanf("%s", response);
    if (cnt!=0 && strcmp(response, "ACK")==0){
        return true;
    }
    return false;
}
