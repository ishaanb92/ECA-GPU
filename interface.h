#ifndef _INTERFACE_H_
#define _INTERFACE_H_
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include "common.h"

//request new input data
bool requestInput(char* hash);

//validate the hash that was supplied
bool validateHash(char* hash, char* nonce);

#endif