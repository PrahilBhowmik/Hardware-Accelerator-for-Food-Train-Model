#pragma once 
#include "./include/k2c_tensor_include.h" 
void foodTrain(size_t input_1_input_ndim,size_t input_1_input_numel,size_t input_1_input_shape[],float input_1_input_array[], 
	size_t activation_output_ndim,size_t activation_output_numel,size_t activation_output_shape[],float activation_output_array[]);
void foodTrain_initialize(); 
void foodTrain_terminate(); 
