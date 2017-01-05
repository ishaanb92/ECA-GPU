#!/usr/bin/python
import random
import subprocess
import string
import time
import re
import os
import sys

#hash settings
inputSize=32
inputSizeMult=4096
nonce_size_chars=1
leading_zeroes=3
#end of hash settings

#benchmark settings
number_of_test_inputs=1000
#end of benchmark settings


if len(sys.argv)==2:
    if sys.argv[1].lower()=='--short':
        inputs=[
        '4RESRCD36URP404HGE8S4HWT835FC598', #valid
        'AS14M842MC771D43G76QLQS98TZN9WS8', #valid
        'QE4I6OPWMOP6M5II7T4TGMOJF4DTPVOP', #valid
        '20Z42WVJPIVAMUNAK9EDYJVRIOHATSUR', #valid
        '00000000000000000000000000000000', #invalid
        'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', #invalid
        'BVNH5YVA9Q5MK01M9J7M9NIS5R4ZLFL6', #invalid
        '1TDNBA3AZFUEZS4HZSZ95TU99CEFL5HZ', #invalid
        ]
    else:
        print 'unknown commandline argument '+sys.argv[1]
        exit(-1)
else:
    #generate number_of_test_inputs inputs
    random.seed(13371337)
    inputs=[''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(inputSize)) for _ in range(number_of_test_inputs)]


def check_hash(result_hash):
    if result_hash[0:leading_zeroes]=='0'*leading_zeroes:
        return True
    return False

def isValidNonce(n):
    if re.match('^[a-zA-Z0-9]+$', str(n))!=None:
        return True
    return False

start_time = time.time()
try:
    #check if we have the proper reference_hash binary
    if not os.path.isfile('./reference_hash'):
        print 'The reference_hash binary could not be found, please make sure it is in the same folder as the benchmark script'
        exit(-1)

    #fix permissions of reference hash if wrong
    if not os.access('./reference_hash', os.X_OK):
        os.system('chmod u+x ./reference_hash')

    #init score to 0
    score=0

    #start executable
    proc = subprocess.Popen(['./miner', '-benchmark'], stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=False)

    #handle commands
    solved=0
    HashIsOut=False
    while solved!=len(inputs):
        #read input from process
        cmd=[s.strip() for s in proc.stdout.readline().split() if s.strip()!='']
        if len(cmd)==0 or len(cmd)>3:
            print 'Your program provided an unsupported command'
            proc.stdin.write('NAK\n')
            continue

        #R              //request new input
        if cmd[0]=='R': #new input
            if HashIsOut:
                print 'Your program is trying to obtain new hash before solving previous...this is not allowed!'
                proc.stdin.write("NAK\n")
                continue

            hashOut=inputs[solved]
            proc.stdin.write(hashOut+'\n')
            HashIsOut=True

        #V hash nonce   //validate hash + nonce ('NONE' if no valid hash was found)
        elif cmd[0]=='V': #validate hash
            if not HashIsOut:
                print'Trying to solve hash but none was requested!'
                proc.stdin.write('NAK\n')
                continue
            HashIsOut=False
            if len(cmd)!=3:
                print"'V' command given with incorrect number of parameters!"
                proc.stdin.write('NAK\n')
                continue
            hash=str(cmd[1])
            if hash!=hashOut:
                print'Trying to validate different hash than what was supplied!'
                proc.stdin.write('NAK\n')
                continue

            nonce=str(cmd[2]).strip()

            if nonce.lower()=='none':
                #print'Your program claimed there are no solutions to given hash (this can be normal behavior if no hash is found!)'
                solved+=1
                proc.stdin.write('ACK\n')
                continue

            if len(nonce)!=nonce_size_chars:
                print 'Provided Nonce is too large!'
                proc.stdin.write('NAK\n')
                continue

            if not isValidNonce(nonce):
                print 'Illegal Nonce provided!'
                proc.stdin.write('NAK\n')
                continue

            else:
                #check hash
                hash_proc = subprocess.Popen(['./reference_hash', str(nonce), str(hash), str(inputSizeMult)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                result_hash= hash_proc.stdout.readline().strip()
                print hash, inputSizeMult, nonce, result_hash
                if check_hash(result_hash):
                    solved+=1
                    score+=1
                    proc.stdin.write('ACK\n')
                    print'Congratulations, you found a valid coin!'
                else:
                    print'Your program supplied an invalid solution!'
                    proc.stdin.write('NAK\n')
                    # round_invalid+=1
        else:
            print'Your program passed an unsupported command'
            proc.stdin.write('NAK\n')

    #all test inputs are done, terminate program
    proc.terminate()
except:
    print'The program closed unexpectedly...'
    exit(0)

#
print 'Processed all inputs'
print '-'*40
print 'Summary:'
print '-'*40
print "%-20s %d"%('- Processed Blocks:',len(inputs))
print "%-20s %d"%('- Number of Coins:', score)
print "%-20s %d sec"%('- Running Time:', time.time() - start_time)
print '-'*40

