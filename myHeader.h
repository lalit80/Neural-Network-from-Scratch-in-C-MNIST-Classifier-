#include <math.h>
#include <stdlib.h>


// function declarations
double sigmoid(double);
double sigmoid_derivative(double);
double rand_uniform();
double rand_uniform_xi(double, double);
void softmax(double*, double*, int);
void copyArray(double*, double*, int);
void multiply_vec(double*, double*, double*, int);
void multiply_to_vec(double*, double, int);
void subtract_vec(double*, double*, double*, int);
double vector_sum(double*, int);
double vector_multiply_sum(double*, double*, int);
double relu(double);
double relu_derivative(double);

// function definitions
double rand_uniform()
{
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;  // [-1, 1]
    //return (double) rand() / RAND_MAX;
}

double rand_uniform_xi(double lower, double upper)
{
    return lower + ((double)rand() / RAND_MAX) * (upper - lower);
}

double sigmoid(double z)
{
    return (1.0 / (1.0 + exp(-z)));
}

double sigmoid_derivative(double z)
{
    double s = 1.0 / (1.0 + exp(-z));
    return s * (1.0 - s);
}

void copyArray(double* dest, double* src, int n)
{
    for (int i = 0; i < n; ++i)
        dest[i] = src[i];
}

void multiply_vec(double* res, double* v1, double* v2, int n)
{
    for (int i = 0; i < n; ++i)
        res[i] = v1[i] * v2[i];
}

void multiply_to_vec(double* v, double d, int n)
{
    for (int i = 0; i < n; ++i)
        v[i] *= d;
}

void subtract_vec(double* res, double* v1, double* v2, int n)
{
    for (int i = 0; i < n; ++i)
        res[i] = v1[i] - v2[i];
}

double vector_multiply_sum(double* v1, double* v2, int n)
{
    double res = 0.0;
    for (int i = 0; i < n; ++i)
        res += (v1[i] * v2[i]);
    return res;
}

double vector_sum(double* v, int n)
{
    double res = 0.0;
    for (int i = 0; i < n; ++i)
        res += v[i];
    return res;
}

void softmax(double* z, double* a, int n)
{
    double max_z = z[0];
    for (int i = 1; i < n; ++i)
        if (z[i] > max_z) max_z = z[i];

    double sum_exp = 0.0;
    for (int i = 0; i < n; ++i) {
        a[i] = exp(z[i] - max_z);
        sum_exp += a[i];
    }

    for (int i = 0; i < n; ++i)
        a[i] /= sum_exp;
}

double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}
