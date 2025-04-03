#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}


void clampedExpVector(float *values, int *exponents, float *output, int N) {
    //
    // PP STUDENTS TODO: Implement your vectorized version of
    // clampedExpSerial() here.
    //
    // Your solution should work for any value of
    // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
    //

    __pp_vec_float threshold = _pp_vset_float(9.999999f);
    __pp_vec_int zeros = _pp_vset_int(0);
    __pp_vec_int ones = _pp_vset_int(1);

    __pp_vec_float result, value;
    __pp_vec_int count;
    __pp_mask maskDone;


    for (int i = 0; i < N; i += VECTOR_WIDTH) {

        // All ones 
        maskDone = _pp_init_ones(VECTOR_WIDTH);

        // load exponents to count
        _pp_vload_int(count, exponents + i, maskDone);

        // load values to value
        _pp_vload_float(value, values + i, maskDone);

        // update mask
        _pp_vgt_int(maskDone, count, zeros, maskDone);

        // set vector to 1.0
        result = _pp_vset_float(1.f);

        // load values to value
        _pp_vload_float(result, values + i, maskDone);
        
        _pp_vgt_int(maskDone, count, ones, maskDone);

        while(_pp_cntbits(maskDone) > 0) {
            _pp_vmult_float(result, result, value, maskDone);
            _pp_vsub_int(count, count, ones, maskDone);
            _pp_vgt_int(maskDone, count, ones, maskDone);
        }

        // All ones
        maskDone = _pp_init_ones(min(VECTOR_WIDTH, N - i));

        // load all result
        _pp_vstore_float(output + i, result, maskDone);

        // replace result that exceed the threshold
        _pp_vgt_float(maskDone, result, threshold, maskDone);
        _pp_vstore_float(output + i, threshold, maskDone);       
    }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N){

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  __pp_mask All = _pp_init_ones(VECTOR_WIDTH);
  __pp_vec_float result, value;

  float temp[VECTOR_WIDTH];
  float ret = 0.f;

  for (int i = 0; i < N; i += VECTOR_WIDTH){
      _pp_vload_float(value, values + i, All);
      int v = VECTOR_WIDTH;

      while(v > 1) {
          _pp_hadd_float(result, value);
          _pp_interleave_float(value, result);
          v /= 2;
      }
      _pp_vstore_float(temp, value, All);
      ret += temp[0];
  }

  return ret;
}