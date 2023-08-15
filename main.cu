#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

__global__ void MatrixAdditionKernel(int *M,int *N,int* A,int p,int r){
	int id= blockIdx.x*blockDim.x+threadIdx.x;
	if(id<p*r){
		A[id]=M[id]+N[id];
	}
	
}


__global__ void MatrixMulKernel(int* M, int* N, int* P, int p1,int q1,int r1 ){
  int TILE_WIDTH=32;
  int Width=q1;
__shared__ int ds_M[32][32];
__shared__ int ds_N[32][32];
if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
 for (int i = 0; i < 32; i ++){
     for(int j=0;j<32;j++){
         ds_M[i][j]=0;
         ds_N[i][j]=0;
     }
 }
  
__syncthreads();
int bx = blockIdx.x; int by = blockIdx.y;
int tx = threadIdx.x; int ty = threadIdx.y;
int Row = by * blockDim.y + ty;
int Col = bx * blockDim.x + tx;
int Pvalue = 0;
 __syncthreads();

for (int p = 0; p < (q1+TILE_WIDTH-1)/TILE_WIDTH; ++p) {
{
if(Row<p1 )
    ds_M[ty][tx] = M[Row*Width + p*TILE_WIDTH+tx];
 
if(Col<r1)
    ds_N[ty][tx] = N[(p*TILE_WIDTH+ty)*r1+Col];

__syncthreads();
 

for (int i = 0; i < TILE_WIDTH; ++i){
    Pvalue += ds_M[ty][i] * ds_N[i][tx];
}
__syncthreads();

}
}
 if(Row<p1 && Col<r1)
    P[Row*r1+Col] = Pvalue;
 

 //printf("%d  %d   %d  %d\n",Row,Col,Row*r1+Col,Pvalue);
 /* if(Row==4 && Col==0 && p==1){
     for(int i=0;i<2;i++){
         for(int j=0;j<2;j++){
             printf("%d ",ds_N[i][j]);
         }
         printf("\n");
     }
 }
 */
}



// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE,*d_matrixE1,*d_matrixE2;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));
	cudaMalloc(&d_matrixE1, p * r * sizeof(int));
	cudaMalloc(&d_matrixE2, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */

	/* ****************************************************************** */
	int threads=32;
    int xblocks= (r+threads-1)/threads;
    int yblocks=(p+threads-1)/threads;
    
    dim3 grid(xblocks,yblocks,1);
    dim3 block(threads,threads,1);
	MatrixMulKernel<<<grid,block>>>(d_matrixA, d_matrixB,d_matrixE1, p,q,r);
	MatrixMulKernel<<<grid,block>>>(d_matrixC, d_matrixD,d_matrixE2, p,q,r);

	int nblocks= (p*r+1024-1)/1024;
	MatrixAdditionKernel<<<nblocks,1024>>>(d_matrixE1,d_matrixE2,d_matrixE,p,r);

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}






// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}





// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}






int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);



	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);



	//taking the transpose of the d matrix
	int *matrixDT=(int*) malloc(r * q * sizeof(int));
	int len=r*q;
	for(int i=0;i<len;i++){
		int row=i/q;
		int col=i%q;
		matrixDT[col*r+row]=matrixD[i];
	}




	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixDT, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);



	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
	
