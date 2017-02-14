#ifndef HEADER_H
#define HEADER_H

void addWithCuda(float *dot, const float *a, const float *b, unsigned int size);
void addWithCudaShared(float *dot, const float *a, const float *b, unsigned int size);


#endif