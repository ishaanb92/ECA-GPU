#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <string.h>

//SETTINGS
#define UNIQUE_INPUT_SIZE   32
#define INPUT_MULT          4096
#define INPUT_SIZE          (UNIQUE_INPUT_SIZE*INPUT_MULT)
#define NONCE_SIZE          1
#define LEADING_ZEROES      3
//END


#define ROR32(x, y)  ((y==0)?x:(((x) >> (y)) | ((x) << (32-(y)))))

uint32_t ROR(uint32_t x, int y);

extern const char _hex[];

//######################################################################################################################
// Helper Functions
//######################################################################################################################
char                 hex(int nibble);
void                 stringtohex_BE(char* in, char* out);

#endif
