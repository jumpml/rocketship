//
//  main.c
//  test_nndsp
//
//  Created by Raghavendra Prabhu on 3/16/22.
//

#include <stdio.h>
#include <stdint.h>

//#include "nnlib.h"
//#include "utils.h"
//#include <math.h>
#include "fixed_point_math.h"

#include "nntests.h"
#include "dsptests.h"
void SLIMIT_TEST(void)
{
    int numBits = 16;
    int32_t x = (1U << (numBits-1)) + 1;
    int16_t x_lim_p =  SLIMIT(x, numBits);
    int16_t x_lim_n =  SLIMIT(-x, numBits);
    printf("x = %x slim(x) = %x \n -x = %x slim(-x) = %x\n", x, x_lim_p,-x, x_lim_n);
}




int main(int argc, const char * argv[]) {
    RUN_NNTESTS();
    RUN_DSPTESTS(); 
    return 0;
}
