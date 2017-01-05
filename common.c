#include "common.h"

const char _hex[] = "0123456789ABCDEF";

uint32_t ROR(uint32_t x, int y){
    int y_mod = ((y & 0x1F) + 32) & 0x1F;
    return ROR32(x, y_mod);
}

char hex(int nibble){
    return _hex[nibble];
}

void stringtohex_BE(char* in, char* out){
    int j=0;
     for(int i=0;i<64;i+=2)
    {
        out[i] = hex((in[j] & 0xF0) >> 4);
        out[i+1]= hex((in[j] & 0x0F) >> 0);
        j++;
    }
    out[64]='\0';
    return;
}