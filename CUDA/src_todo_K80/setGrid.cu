
#include "mytypes.h"
#include <stdio.h>

void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{

   // set your block dimensions and grid dimensions here
   if(n != 256){
      gridDim.x = n / (TL * 2);
      gridDim.y = n / TW;
   }
   else{
      gridDim.x = n / (TW * 4);
      gridDim.y = n / TW;
   }

}
