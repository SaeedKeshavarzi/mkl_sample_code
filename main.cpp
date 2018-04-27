#include <stdio.h>
#include <chrono>

#include <mkl.h>

#define MAT_SIZE 300
#define MAT_EL_COUNT (MAT_SIZE * MAT_SIZE)
#define N_ITERATE 10000

int main()
{
	int ret = 0;

	double *A, *B, *C;
	double alpha{ 1. }, beta{ 0. };

	A = (double *)mkl_malloc(MAT_EL_COUNT * sizeof(double), 64);
	B = (double *)mkl_malloc(MAT_EL_COUNT * sizeof(double), 64);
	C = (double *)mkl_malloc(MAT_EL_COUNT * sizeof(double), 64);
	if (A == NULL || B == NULL || C == NULL)
	{
		printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");

		mkl_free(A);
		mkl_free(B);
		mkl_free(C);

		ret = 1;
		goto SAY_GOODBYE;
	}

	srand(std::chrono::steady_clock::now().time_since_epoch().count());
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < MAT_EL_COUNT; ++j)
		{
			A[j] = (rand() % MAT_SIZE) / (MAT_SIZE * 0.5);
			B[j] = (rand() % MAT_SIZE) / (MAT_SIZE * 0.5);
		}

		auto _1 = std::chrono::steady_clock::now();

		for (int k = 0; k < N_ITERATE; ++k)
		{
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				MAT_SIZE, MAT_SIZE, MAT_SIZE, alpha, A, MAT_SIZE, B, MAT_SIZE, beta, C, MAT_SIZE);
		}

		auto _2 = std::chrono::steady_clock::now();
		auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(_2 - _1).count();

		printf("%lld ms, %lld multiply/sec \n", diff, (N_ITERATE * 1000) / diff);
	}

	printf("\n Deallocating memory \n\n");
	mkl_free(A);
	mkl_free(B);
	mkl_free(C);

	printf(" Example completed. \n\n");

SAY_GOODBYE:
	printf(" press any key to continue... ");
	getchar();
	return ret;
}
