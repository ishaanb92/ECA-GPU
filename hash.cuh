#include <stdint.h>
#include "common.h"

typedef uint8_t u8;
typedef uint32_t u32;

//hash settings
#define CRYPTO_BYTES (64)
#define ROUNDS     (14)
#define STATEBYTES (128)
#define STATEWORDS (STATEBYTES/4)
#define STATECOLS  (STATEBYTES/8)
#define COLWORDS     (STATEWORDS/8)
#define BYTESLICE(i) (((i)%8)*STATECOLS+(i)/8)

#if CRYPTO_BYTES<=32
__device__ static const u32 columnconstant[2] = { 0x30201000, 0x70605040 };
__device__ static const u8 shiftvalues[2][8] = { {0, 1, 2, 3, 4, 5, 6, 7}, {1, 3, 5, 7, 0, 2, 4, 6} };
#else
__device__ static const u32 columnconstant[4] = { 0x30201000, 0x70605040, 0xb0a09080, 0xf0e0d0c0 };
__device__ static const u8 shiftvalues[2][8] = { {0, 1, 2, 3, 4, 5, 6, 11}, {1, 3, 5, 11, 0, 2, 4, 6} };
#endif

__device__ static const u8 S[256] = {
  0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
  0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
  0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
  0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
  0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
  0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
  0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
  0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
  0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
  0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
  0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
  0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
  0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
  0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
  0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
  0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
  0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
  0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
  0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
  0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
  0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
  0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
  0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
  0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
  0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
  0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
  0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
  0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
  0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
  0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
  0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
  0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

__device__ unsigned char nonce_array[62]={'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
        '0','1','2','3','4','5','6','7','8','9'};

#define mul2(x,t) \
{\
  t = x & 0x80808080;\
  x ^= t;\
  x <<= 1;\
  t = t >> 7;\
  t ^= (t << 1);\
  x ^= t;\
  x ^= (t << 3);\
}

__device__ void mixbytes(u32 a[8][COLWORDS], u32 b[8], int s)
{
  __shared__ int i;
  __shared__ u32 t0, t1, t2;

  for (i=0; i<8; i++)
    b[i] = a[i][s];

  /* y_i = a_{i+6} */
  for (i=0; i<8; i++)
    a[i][s] = b[(i+2)%8];

  /* t_i = a_i + a_{i+1} */
  for (i=0; i<7; i++)
    b[i] ^= b[(i+1)%8];
  b[7] ^= a[6][s];

  /* y_i = a_{i+6} + t_i */
  for (i=0; i<8; i++)
    a[i][s] ^= b[(i+4)%8];

  /* y_i = y_i + t_{i+2} */
  for (i=0; i<8; i++)
    a[i][s] ^= b[(i+6)%8];

  /* x_i = t_i + t_{i+3} */
  t0 = b[0];
  t1 = b[1];
  t2 = b[2];
  for (i=0; i<5; i++)
    b[i] ^= b[(i+3)%8];
  b[5] ^= t0;
  b[6] ^= t1;
  b[7] ^= t2;

  /* z_i = 02 * x_i */
  for (i=0; i<8; i++)
    mul2(b[i],t0);

  /* w_i = z_i + y_{i+4} */
  for (i=0; i<8; i++)
    b[i] ^= a[i][s];

  /* v_i = 02 * w_i */
  for (i=0; i<8; i++)
    mul2(b[i],t0);

  /* b_i = v_{i+3} + y_{i+4} */
  for (i=0; i<8; i++)
    a[i][s] ^= b[(i+3)%8];
}

__device__ void permutation(u32 *x, int q)
{
 __shared__ __align__(8) u32 tmp[8];
  __shared__ u32 constant;
  __shared__ int i;
  uint32_t tx = threadIdx.x;
  for(constant=0; constant<(0x01010101*ROUNDS); constant+=0x01010101)
  {
    if (q==0)
    {
      if (tx < COLWORDS)
        x[tx] ^= columnconstant[tx]^constant;
      __syncthreads();
    }
    else
    {
      x[tx] = ~x[tx];
      __syncthreads();
      if (tx < COLWORDS)
        x[STATEWORDS-COLWORDS+tx] ^= columnconstant[tx]^constant;
      __syncthreads();
    }
    for (i=0; i<8; i++) {
      if (tx < COLWORDS)
        tmp[tx] = x[i*COLWORDS+tx];
      __syncthreads();
      
      if (tx < STATECOLS)
        ((u8*)x)[i*STATECOLS + tx] = S[((u8*)tmp)[(tx +shiftvalues[q][i])%STATECOLS]];
      __syncthreads();
    }

    if (tx < COLWORDS)
      mixbytes((u32(*)[COLWORDS])x, tmp, tx);
    __syncthreads();
  }
}

__device__ void memxor(u32* dest, const u32* src, u32 n)
{
    dest[threadIdx.x] ^= src[threadIdx.x];
}

struct state {
  u8 bytes_in_block;
  u8 first_padding_block;
  u8 last_padding_block;
};

__device__ void setmessage(u8* buffer, const u8* in, struct state s, unsigned long long inlen)
{
  __shared__ int i;
  uint32_t tx = threadIdx.x;

  i = s.bytes_in_block;

  if (!s.first_padding_block && !s.last_padding_block) { // rlen > STATEBYTES, s.bytes_in_block = STATEBYTES
    for (uint32_t j = 0; j < STATEBYTES/32 ; j ++)
      buffer[BYTESLICE(j*32 + tx)] = in[j*32 + tx];
  __syncthreads();
  }
  else { // s.bytes_in_block = 1
    
    buffer[BYTESLICE(0)] = in[0];

    if (s.first_padding_block)
    {
      buffer[BYTESLICE(i)] = 0x80;
      i++;
    }

    for(;i<STATEBYTES;i++)
      buffer[BYTESLICE(i)] = 0;

    if (s.last_padding_block)
    {
      inlen /= STATEBYTES;
      inlen += (s.first_padding_block==s.last_padding_block) ? 1 : 2;
      if (tx < 8)
        buffer[BYTESLICE(tx + (STATEBYTES-8))] = (inlen >> 8*(STATEBYTES-(tx +(STATEBYTES-8)) -1)) & 0xff;
    }
    __syncthreads();
  }

}


__device__ bool check_hash(char* hash){
    //check if first n-characters are zero
    for(int i=0;i<LEADING_ZEROES;i++)
		//Note: each 'char' of 8 bits contains 2 hex characters representing 4 bits each.
		//Hence all this bit shuffling
		if ((hash[i>>1]&(0xF0>>((i&0x1)<<2)))!=0)
            return false;
    return true;
}

__global__ void hash(char *nonce, unsigned char *in)
{

  __shared__ __align__(8) u32 ctx[STATEWORDS];
  __shared__ __align__(8) u32 buffer[STATEWORDS];

  __shared__ unsigned long long inlen; 
  __shared__ unsigned long long rlen;
  inlen = INPUT_MULT*UNIQUE_INPUT_SIZE + NONCE_SIZE;
  rlen = inlen;

  __shared__ unsigned char base[UNIQUE_INPUT_SIZE];
  
  uint32_t bx = blockIdx.x;
  uint32_t tx = threadIdx.x;
  
  unsigned char *in_ptr; // per block input ptr

  in_ptr = &(in[bx*(INPUT_SIZE+NONCE_SIZE)]);

  base[tx] = in[tx+1]; // Copy base string per block
  __syncthreads(); 
#pragma unroll
  for (uint32_t i = 0; i < INPUT_MULT; i ++) {
    in_ptr[i*UNIQUE_INPUT_SIZE + tx + NONCE_SIZE] = base[tx]; // Copy for each block
  }

  // For each block, assign a unique a nonce
  in_ptr[0] = nonce_array[bx];


  __shared__ char out[64]; // Output hash

  __shared__ struct state s;

  s.bytes_in_block = STATEBYTES;
  s.first_padding_block = 0;
  s.last_padding_block = 0;

  __shared__ u8 i;

  // Declare a pointer to block_input

  /* set inital value */
  
  ctx[tx] = 0;
  __syncthreads();

  if (tx < 1) {
    ((u8*)ctx)[BYTESLICE(STATEBYTES-2)] = ((CRYPTO_BYTES*8)>>8)&0xff;
    ((u8*)ctx)[BYTESLICE(STATEBYTES-1)] = (CRYPTO_BYTES*8)&0xff;
  }
  __syncthreads();

  /* iterate compression function */
  while(s.last_padding_block == 0)
  {
    if (rlen<STATEBYTES)
    {
      if (s.first_padding_block == 0)
      {
        s.bytes_in_block = rlen;
        s.first_padding_block = 1;
        s.last_padding_block = (s.bytes_in_block < STATEBYTES-8) ? 1 : 0;
      }
      else
      {
        s.bytes_in_block = 0;
        s.first_padding_block = 0;
        s.last_padding_block = 1;
      }
    }
    else
      rlen-=STATEBYTES;

    /* compression function */
    setmessage((u8*)buffer, in_ptr , s, inlen);
    __syncthreads();
    memxor(buffer, ctx, STATEWORDS);
    permutation(buffer, 0);
    memxor(ctx, buffer, STATEWORDS);
    setmessage((u8*)buffer, in_ptr, s, inlen);
    __syncthreads();
    permutation(buffer, 1);
    memxor(ctx, buffer, STATEWORDS);

    /* increase message pointer */
    in_ptr  += STATEBYTES;
  }

  /* output transformation */
  buffer[tx] = ctx[tx];
  __syncthreads();

  permutation(buffer, 0);
  memxor(ctx, buffer, STATEWORDS);

  /* return truncated hash value */
#if 0
#pragma unroll
  for (i = STATEBYTES-CRYPTO_BYTES; i < STATEBYTES; i++)
    out[i-(STATEBYTES-CRYPTO_BYTES)] = ((u8*)ctx)[BYTESLICE(i)];
#endif
    for (i = 0; i < 2; i++)
      out[tx + i*(STATEBYTES-CRYPTO_BYTES)] = ((u8*)ctx)[BYTESLICE(tx + i*(STATEBYTES - CRYPTO_BYTES) + (STATEBYTES-CRYPTO_BYTES))];
    __syncthreads();

/*
  // Copy the per-block output
  for (uint32_t k=0; k < 64; k++)
    output[bx*64 + k] = out[k];
*/


  // For each block check if output hash has a coin
  if (tx < 1) {
    if (check_hash(&(out[0]))) {
      nonce[0] = nonce_array[bx]; // Assign the winning nonce
    }
  }
  __syncthreads();

}
