#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <mpi.h>

#include <sys/stat.h>
#include <sys/time.h>

#define ROOT 0
#define MAX_THREAD 512
#define MAXNGPU 3	//maximum number of GPU that can be used
#define M 200		// number of starting points per iteration
#define ITERMAX 200	//maximum number of moot iterations in a row after which the program would stop

#define BASE 5
#define POWER 13
//#define PRIME (pow(BASE,POWER))
#define DP 21


#define TIMER_DEF struct timeval temp_1, temp_2
#define TIMER_START gettimeofday(&temp_1, (struct timezone*)0)
#define TIMER_STOP gettimeofday(&temp_2, (struct timezone*)0)
#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)*1.e6+(temp_2.tv_usec-temp_1 .tv_usec))

//instructions for compiling and running
//nvcc -I/usr/local/openmpi-4.1.4/include -L/usr/local/openmpi-4.1.4/lib -lmpi base5DP.cu -o baseDP
// mpirun -np 7 baseDP 503 11

void cudaErrorCheck(cudaError_t error, const char * msg){
   	 if ( error != cudaSuccess){
   	 fprintf(stderr, "%s:%s\n ", msg, cudaGetErrorString(error));
   	 exit(EXIT_FAILURE);}}

//subfunction to be called inside baseFunct
__host__ __device__ uint32_t abcFunct(uint32_t ua, uint32_t ub, uint32_t uc){
    	int a=ua;
    	int b=ub;
    	int c=uc;
    	uint32_t F=c+pow(a-b,2);
    	return F;
}

//function that advances steps in the path
__host__ __device__ uint32_t baseFunct(uint32_t x, int r){
    	r=r%(POWER+1);
    	uint32_t *arrayEquivX=(uint32_t*)malloc(sizeof(int)*(POWER+r));
    	uint32_t exp= pow(BASE, POWER);
    	uint32_t remnant;
    	uint32_t y;

    	remnant=x;

	//padding for the indexes out of bound
    	for (int i=0; i<r; i++) arrayEquivX[POWER+i]=0;

    	for (int i=0; i<POWER; i++){
            	exp=exp/BASE;
            	arrayEquivX[POWER-i-1]=remnant/exp;
            	remnant=remnant%exp;
            	//printf("in pos %d (exp %d) there is %d with remainder %d \n", POWER-1-i, exp, arrayEquivX[POWER-i-1],remnant);
    	}

    	y=0;

    	for (int i=0; i<POWER; i++){
            	//arrayEquivX[i]= abcFunct(arrayEquivX[i], arrayEquivX[i+1], arrayEquivX[i+j]);
            	//printf("(using %d) in pos %d there is %d \n", arrayEquivX[i+r], i, arrayEquivX[i]);
            	y=y+exp*abcFunct(arrayEquivX[i], arrayEquivX[i+1], arrayEquivX[i+r]);
            	exp=exp*BASE;
    	}
    	//printf("modulo is %d \n", exp);
    	y=y%exp;
    	return y;
}


//computes path on the GPU
__global__ void findPathAndDP(
   	uint32_t* d_x0p,
   	uint32_t* d_x,
	int lg,
    	int r,
   	int n_per_proc,
   	uint32_t* d_DjX0,
   	uint32_t* d_DjD,
   	uint32_t* d_Djsteps,
   	uint32_t* d_DjC,
   	int* d_nDPaux
) {

   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   	 
   if (tid<n_per_proc) {
   	//initialization of arrays
   	d_nDPaux[tid]=0;
   	d_DjX0[tid]=0;
   	d_DjD[tid]=0;
   	d_Djsteps[tid]=0;
   	d_DjC[tid]=0;
   	
	//places starting points in the right position
   	d_x[tid*lg] = d_x0p[tid];
       	//printf("starting point %d \n", d_x[tid*lg]);
       	 __syncthreads();
   	
	//all threads start computing path
       	for(int i = 0; i < lg-1; i++) {
               	//d_x[tid*lg+i+1] = (d_x[tid*lg+i]*d_x[tid*lg+i]+1) % PRIME;
   		d_x[tid*lg+i+1]=baseFunct(d_x[tid*lg+i],r);
               	//printf("indx %d path ID %d, %d step %d \n",tid*lg+i, tid, i+1, d_x[tid*lg+i+1]);
               	
		//finds DPs
   		if (d_x[tid*lg+i+1] % DP == 0) {
                	d_nDPaux[tid]=1;
                       	d_DjX0[tid] = d_x0p[tid];
                       	d_DjD[tid] = d_x[tid*lg+i+1];
                       	d_Djsteps[tid] = i+1;
                       	d_DjC[tid] = d_x[tid*lg+i];
                       	//printf("DP found %d in indx %d \n", d_DjD[tid], tid);
                       	break;
               		}
   	 }
       	//__syncthreads();       		 
   	 }
}



//subRoutine to find collisions that happen inside (rather than on last step) of the path
void FindIntermediateColl (int r, uint32_t DjX0i, uint32_t Djstepsi,
   		 uint32_t DjX0k, uint32_t Djstepsk, uint32_t* newDjC, uint32_t* newDjD){

	//printf("in %d steps from %d and %d steps from %d we reach %d \n", Djsteps[i], DjX0[i], Djsteps[k], DjX0[k], DjD[i]);
        int diff;
        int lim;
                   	 
   	if (Djstepsi<Djstepsk){
                diff=Djstepsk-Djstepsi;
                lim=Djstepsi;
                }
        else{
                diff=Djstepsi-Djstepsk;
                lim=Djstepsk;
                }
   		 
   	uint32_t *tempReach= (uint32_t*)malloc(sizeof(int)*(diff+1));
        uint32_t *tempShort= (uint32_t*)malloc(sizeof(int)*lim);
        uint32_t *tempLong=  (uint32_t*)malloc(sizeof(int)*lim);
                   	 
   	if (Djstepsi<Djstepsk){
                tempShort[0]=DjX0i;
                tempReach[0]=DjX0k;
                }
        else{
                tempShort[0]=DjX0k;
                tempReach[0]=DjX0i;
                }

        for (int d=0; d<diff; d++) {
        tempReach[d+1]=baseFunct(tempReach[d],r);
   	//printf("%d step %d \n", d, tempReach[d]);
        }

	//checks where the two paths are at the same distance (from the end) to the DP they both converge on
        tempLong[0]=tempReach[diff];
        //printf (" %d step %d \n", diff, tempLong[0]);

        if (tempShort[0]!=tempLong[0]){
        	//printf("%d and %d will collide on %d in %d steps \n", tempShort[0], tempLong[0], DjD[i], lim);
        	for(int l=0; l<lim-1;l++){
                        tempShort[l+1]=baseFunct(tempShort[l],r);
                        tempLong[l+1] =baseFunct(tempLong[l],r);
                        if (tempShort[l+1]==tempLong[l+1]){
                                newDjC[0]=tempShort[l];
                                newDjC[1]=tempLong[l];
                                newDjD[0]=tempShort[l+1];
                                newDjD[1]=tempLong[l+1];
                                break;
                                }
                }
                printf("intermediate collision between %d and %d on %d \n", newDjC[0], newDjC[1], newDjD[0]);
        }
        //Free temporary Array
	free(tempReach);
        free(tempShort);
        free(tempLong);
}


    
int main (int argc, char** argv) {

TIMER_DEF;
TIMER_START;

//input validation
if(argc != 3){
    fprintf(stderr,"wrong number of inputs\n");
    return EXIT_FAILURE;}

int lg=atoi(argv[1]);

if(lg <=0){
   	 fprintf(stderr,"[ERROR] - lg must be > 0\n");
   	 return EXIT_FAILURE;}

//index to use inside "stepping" function
int r=atoi(argv[2]);

 if(r <0){
   	 fprintf(stderr,"[ERROR] - r must be > 0\n");
   	 return EXIT_FAILURE;}

uint32_t PRIME= pow(BASE, POWER);

//MPI initialization
int rank, NP;
MPI_Init(&argc, &argv);

MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &NP);


int n_per_proc; // elements per process
n_per_proc=M/NP;

if (rank==ROOT) printf("num start point per proc %d \n", n_per_proc);

//each process selects a GPU to work on
int usableGPUs;
cudaErrorCheck(cudaGetDeviceCount(&usableGPUs),"cudaGetDevice");
//if (usableGPUs>MAXNGPU) usableGPUs=MAXNGPU;
//printf("%d GPUs available\n", usableGPUs);

//to make performance more efficient, each GPU can oly be assigned one process
if(NP>usableGPUs){
    	fprintf(stderr,"[ERROR] - rerun with less than %d processes\n", usableGPUs);
    	return EXIT_FAILURE; }

cudaErrorCheck(cudaSetDevice(rank%MAXNGPU),"cudaSetDevice");

int *x0glob=(int*)malloc(sizeof(int)*PRIME);
if(x0glob==NULL){
    	fprintf(stderr,"[ERROR] - Cannot allocate memory\n");
    	return EXIT_FAILURE; }

for (int i=0; i<M ; i++) x0glob[i]=0;

//initialization global values and arrays
int nCollisFinal=0;

int *scA=(int*)malloc(sizeof(int)*M*(M-1)/2);
if(scA==NULL){
   	 fprintf(stderr,"[ERROR] - Cannot allocate memory\n");
   	 return EXIT_FAILURE; }
int *scB=(int*)malloc(sizeof(int)*M*(M-1)/2);
for (int i=0; i<M*(M-1)/2 ; i++){
   	 scA[i]=0;
   	 scB[i]=0;}


int nCovered=0;	//number of different starting points used
int nIter=0;	//number of iterations
int nMoot=0;	//number of unuseful iterations in a row

while (nCollisFinal < 1){

	//generation and scattering of unique random starting points
	int nDPj=0;   

	//needs to be allocated on all processes otherwise we can't use the scatter
   	uint32_t *x0=(uint32_t*)malloc(sizeof(int)*M);

    	if (x0==NULL) {
   		 fprintf(stderr,"[ERROR][RANK %d] Cannot allocate memory\n",rank);
   		 MPI_Abort(MPI_COMM_WORLD,1);}
    
    if (rank == ROOT){
    for (int i = 0; i < NP*n_per_proc; i++){
       		 int a=1;
   	 while (a==1){
		//generation
		x0[i]=rand()%PRIME;
       		//printf("indx %d-  %d \n", i, x0[i]);
       		int b=0;
		//checks that different from all other starting points of this iteration
   		for(int k=0; k<i; k++){
   		if (x0[i]==x0[k]){
   			b++;
   			break;}}
   		 if (b==0) a=0;
   		 }//WHILE
   	int c=0;
	//checks if starting has not been used in previous iterations and adds to global count
   	for(int k=0; k<nCovered; k++){
   	if (x0[i]==x0glob[k]){
   		c++;
           	break;}}
   	if (c==0){
   		x0glob[nCovered]=x0[i];
   		nCovered++;}
    	}//FOR
    }//ROOT

   	 uint32_t *x0p=(uint32_t*)malloc(sizeof(int)*n_per_proc);

   	 if(x0p==NULL){
   	 fprintf(stderr,"[ERROR] - Cannot allocate memory\n");
   	 return EXIT_FAILURE; }

   	MPI_Bcast(&nCovered,1, MPI_INT,ROOT,MPI_COMM_WORLD);
	MPI_Scatter(x0, n_per_proc, MPI_INT, x0p, n_per_proc, MPI_INT, ROOT, MPI_COMM_WORLD);
   	MPI_Barrier(MPI_COMM_WORLD);
    
    	free(x0);

    	if(rank==ROOT) printf("scatter success %d \n", M);
   	 
    	//allocation and initialization of device arrays
   	uint32_t *d_x0p;
   	cudaErrorCheck(cudaMalloc(&d_x0p,sizeof(int)*n_per_proc),"cudaMalloc d_x0p");
    	cudaErrorCheck(cudaMemcpy(d_x0p,x0p,sizeof(int)*n_per_proc,cudaMemcpyHostToDevice),"Memcpy d_x0p");

   	uint32_t *d_x;
   	cudaErrorCheck(cudaMalloc(&(d_x), sizeof(int) * lg*n_per_proc),"cudaMalloc d_x");

   	uint32_t *d_DjX0;
   	uint32_t *d_DjD;
   	uint32_t *d_Djsteps;
   	uint32_t *d_DjC;
   	int *d_nDPaux;

   	cudaErrorCheck(cudaMalloc(&(d_DjX0), sizeof(int) *n_per_proc),"cudaMalloc d_DjX0");
   	cudaErrorCheck(cudaMalloc(&(d_DjD), sizeof(int)*n_per_proc),"cudaMalloc d_DjD");
   	cudaErrorCheck(cudaMalloc(&(d_Djsteps), sizeof(int) *n_per_proc),"cudaMalloc d_Djsteps");
   	cudaErrorCheck(cudaMalloc(&(d_DjC), sizeof(int) *n_per_proc),"cudaMalloc d_DjC");
   	cudaErrorCheck(cudaMalloc(&(d_nDPaux), sizeof(int)*n_per_proc),"cudaMalloc d_nDPaux");

    	uint32_t *DjX0=(uint32_t*)malloc(sizeof(int)*n_per_proc);
   	uint32_t *DjD=(uint32_t*)malloc(sizeof(int)*n_per_proc);
   	uint32_t *Djsteps=(uint32_t*)malloc(sizeof(int)*n_per_proc);
   	uint32_t *DjC=(uint32_t*)malloc(sizeof(int)*n_per_proc);
   	int *nDPaux=(int*)malloc(sizeof(int)*n_per_proc);

    	//invocation of CUDA function
    	int nthreads=MAX_THREAD;
   	int nblocks=  n_per_proc/MAX_THREAD+1 ;
   	 
   	findPathAndDP<<<nblocks, nthreads>>>(d_x0p, d_x,lg, r, n_per_proc, d_DjX0, d_DjD, d_Djsteps, d_DjC, d_nDPaux);
                        		              	 

    	//copies from device to host
   	cudaErrorCheck(cudaMemcpy(DjX0, d_DjX0,sizeof(int)*n_per_proc, cudaMemcpyDeviceToHost), "Memcpy");
   	cudaErrorCheck(cudaMemcpy(DjD, d_DjD,sizeof(int)*n_per_proc, cudaMemcpyDeviceToHost), "Memcpy");
   	cudaErrorCheck(cudaMemcpy(Djsteps, d_Djsteps,sizeof(int)*n_per_proc, cudaMemcpyDeviceToHost), "Memcpy");
   	cudaErrorCheck(cudaMemcpy(DjC, d_DjC,sizeof(int)*n_per_proc, cudaMemcpyDeviceToHost), "Memcpy");
   	cudaErrorCheck(cudaMemcpy(nDPaux, d_nDPaux,sizeof(int)*n_per_proc, cudaMemcpyDeviceToHost), "Memcpy");

    	//frees on device
    	cudaFree(d_x0p);
    	cudaFree(d_x);
    	cudaFree(d_x0p);
    	cudaFree(d_DjX0);
    	cudaFree(d_DjD);
    	cudaFree(d_Djsteps);
    	cudaFree(d_DjC);
    	cudaFree(d_nDPaux);

    	//frees on host
    	free(x0p);

    	//calculates tot number of DP per process
   	for (int i=0; i<n_per_proc; i++) nDPj+=nDPaux[i];
    	printf("rank %d - tot DPs found in iteration %d \n", rank, nDPj);
    
    	//flags processes that didn't find any DP
    	int flag=0;
    	if(nDPj==0 && rank!=ROOT) flag=1;
    
    	//finds collisions (per process)
    	int nCollisj=0;
   	uint32_t *CjA=(uint32_t*)malloc(sizeof(int)*n_per_proc*(n_per_proc-1)/2);
   	uint32_t *CjB=(uint32_t*)malloc(sizeof(int)*n_per_proc*(n_per_proc-1)/2);

   	int ncjaux=0;
   	for (int i = 0; i < n_per_proc; i++){
   	for (int k = i+1; k<n_per_proc; k++){
   	if(DjD[i]==DjD[k]){	//if 2 paths converge on the same DP
   		if (DjC[i]==DjC[k]){	//if the collision is not immedate (ie - occuring on the last step) we need to call the relevant function to find it inside the path

			//setup of the Array and Values to pass to the subFunction
			uint32_t *newDjC=(uint32_t*)malloc(sizeof(int)*2);
   			uint32_t *newDjD=(uint32_t*)malloc(sizeof(int)*2);
   		 
	   		newDjC[0]=DjC[i];
	   		newDjC[1]=DjC[k];
	   		newDjD[0]=DjD[i];
	   		newDjD[1]=DjD[k];
	   		 
	   		FindIntermediateColl (r, DjX0[i], Djsteps[i],
	                   			 DjX0[k], Djsteps[k], newDjC, newDjD);
	
			//update "DP"arrays with relevant results from subFunction
	   		DjC[i]=newDjC[0];
	   		DjC[k]=newDjC[1];
	   		DjD[i]=newDjD[0];
	   		DjD[k]=newDjD[1];
	
			//free temporary Array used for subFunction
	   		free(newDjC);
	   		free(newDjD);
   	 	}

   	 if (DjC[i]!=DjC[k]){ //if there is a Collision (ie a!=b| F(a)=F(b) )
       		printf("rank %d collision between %d and %d on %d on indx %d \n", rank, DjC[i], DjC[k], DjD[i], i);

		//we keep the two elements ordered so that is easier to check for duplicates
		if (DjC[i]<DjC[k]){
               		CjA[ncjaux]=DjC[i];
               		CjB[ncjaux]=DjC[k];}
       		else{
               		CjA[ncjaux]=DjC[k];
               		CjB[ncjaux]=DjC[i];}
       			ncjaux++;
       			}
       		}
   	 }} //for i for k 

   	 if ( ncjaux!=0) printf("rank %d - no of collisions %d \n" , rank, ncjaux);    

	//eliminates duplicates (per process)
	//allocates new Array to hold only unique values
   	 uint32_t *scjA=(uint32_t*)malloc(sizeof(int)*n_per_proc*(n_per_proc-1)/2);
   	 uint32_t *scjB=(uint32_t*)malloc(sizeof(int)*n_per_proc*(n_per_proc-1)/2);
   	 for (int i=0; i<n_per_proc*(n_per_proc-1)/2 ; i++){
   	 scjA[i]=0;
   	 scjB[i]=0;}


   	 if (ncjaux>0){
       		 nCollisj=1;
       		 scjA[0]=CjA[0];
       		 scjB[0]=CjB[0];
       		 printf("rank %d -first collis %d and %d \n", rank, scjA[0], scjB[0]);

       		 for (int i = 1; i < ncjaux; i++){
       		 int a=0;
               		 for (int k = 0; k<i; k++){
               		 if(CjA[i]==CjA[k] && CjB[i]==CjB[k]){
                       		 a++;
                       		 break;}}
               		 if (a==0){
   			 scjA[nCollisj]=CjA[i];
   			 scjB[nCollisj]=CjB[i];
                       		 printf("rank %d -collis bw %d and %d \n", rank, scjA[nCollisj], scjB[nCollisj]);
                       		 nCollisj++;}
       		 }
       		 printf("rank %d -no of unique collisions in iteration %d \n" , rank, nCollisj);
   	 }

   	 int nCollisT=0; 	 
    
    //each process shares with root number of collisions found and related information
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&nCollisj, &nCollisT, 1, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);
    
    if (rank==ROOT) printf("reduce success %d \n", nCollisT);

    //initialization and allocation of "gathering" Arrays
    uint32_t *CA=(uint32_t*)malloc(sizeof(int)*M*(n_per_proc-1)/2);
    uint32_t *CB =(uint32_t*)malloc(sizeof(int)*M*(n_per_proc-1)/2);

    if (CA==NULL) {
    	fprintf(stderr,"[ERROR][RANK %d] Cannot allocate memory\n",rank);
    	MPI_Abort(MPI_COMM_WORLD,1);}
    
    for (int i=0; i<M*(n_per_proc-1)/2 ; i++){
    	CA[i]=0;
    	CB[i]=0;}
    
    MPI_Gather(scjA, n_per_proc*(n_per_proc-1)/2, MPI_INT, CA, n_per_proc*(n_per_proc-1)/2, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Gather(scjB, n_per_proc*(n_per_proc-1)/2, MPI_INT, CB, n_per_proc*(n_per_proc-1)/2, MPI_INT, ROOT, MPI_COMM_WORLD);

    //each process shares with root the number of DPs found
    int nDP=0;
    MPI_Reduce(&nDPj, &nDP, 1, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    //each prcess that found DPs>0 shares with root the related information

    //initialization and allocation of "gathering" Arrays    
    uint32_t *DX0=(uint32_t*)malloc(sizeof(int)*M);
    uint32_t *DD =(uint32_t*)malloc(sizeof(int)*M);
    uint32_t *Dsteps =(uint32_t*)malloc(sizeof(int)*M);
    uint32_t *DC =(uint32_t*)malloc(sizeof(int)*M);

	
    for (int i=0; i<M ; i++){
    	DX0[i]=0;
    	DD[i]=0;
    	Dsteps[i]=0;
    	DC[i]=0;}

    //Split the global communicator
    int key;    
    if (flag==0) key= rank;
    else key=NP-rank;
    MPI_Comm new_comm;
    MPI_Comm_split(MPI_COMM_WORLD, flag, key, &new_comm);

    //only processes belonging to new communicator (ie that found DPs) share with root the related information
    MPI_Gather(DjX0, n_per_proc, MPI_INT, DX0, n_per_proc, MPI_INT, ROOT, new_comm);
    MPI_Gather(DjD, n_per_proc, MPI_INT, DD, n_per_proc, MPI_INT, ROOT, new_comm);
    MPI_Gather(Djsteps, n_per_proc, MPI_INT, Dsteps, n_per_proc, MPI_INT, ROOT, new_comm);
    MPI_Gather(DjC, n_per_proc, MPI_INT, DC , n_per_proc, MPI_INT, ROOT, new_comm);

    //frees new communicator
    MPI_Comm_free(&new_comm);

    MPI_Barrier(MPI_COMM_WORLD);

    //frees on device    
   	 free(DjX0);
   	 free(DjD);
   	 free(Djsteps);
   	 free(DjC);
   	 free(scjA);
   	 free(scjB);

    //eliminates duplicates (globally)
    int nCollisTot=0;
    if(rank==ROOT){
    	printf("Cumulative Collis till now %d \n", nCollisFinal);

   	if (nCollisT>0){
       		for (int i = 0; i < M*(n_per_proc-1)/2; i++){
       			int a=0;
       		 	int b=0; 
	   	 	if (CB[i]!=0){
	           		 for (int k = 0; k<i; k++){
	               		 if(CA[i]==CA[k] && CB[i]==CB[k]){	//checks that unique inside iteration
	                       		a++;
	                       		break;}}
	               		//printf( "a= %d on indx %d \n", a, i);
	   		 	if (a==0){	//checks that unique compared to previous iterations
	                       		for (int h = 0;h<nCollisFinal+1;h++){
	                       		if(CA[i]==scA[h] && CB[i]==scB[h]){
	                               		b++;
	                               		break;}}
	                       		 if (b==0){	//if unique adds to global-all-iterations array
	                               		scA[nCollisFinal+nCollisTot]=CA[i];
	                               		scB[nCollisFinal+nCollisTot]=CB[i];
	                               		printf("new Collis bw %d and %d on indx %d \n", scA[nCollisFinal+nCollisTot], scB[nCollisFinal+nCollisTot], i);
	                               		nCollisTot++;}
	   		 		}//if a==0  	 
       		 	} //if CB!=0
		} //for i
       		printf("nSingleRank of this iteration %d \n", nCollisTot);
   	 }
    
    //looks for new "interrank" collsions
    int nCollisIr=0;
    int nCollisIrT=0;
    uint32_t *tempA=(uint32_t*)malloc(sizeof(int)*M*n_per_proc*(NP-1)/2);
    uint32_t *tempB =(uint32_t*)malloc(sizeof(int)*M*n_per_proc*(NP-1)/2);

    for (int i = 0; i <M; i++){
    if (DC[i]==0 && DD[i]!=1) break;
    for (int k =n_per_proc+i/(n_per_proc); k<M; k++){
   	int a=1;
   	int b=1;
   	 
   	if(DC[k]==0 && DD[k]!=1) break;
   	if(DD[i]==DD[k]){
       		a=0;
   		b=0;}
   	if (a==0){
   		if (DC[i]==DC[k]){ //find Intermediate collision routine and setup
                   	uint32_t *newDjC=(uint32_t*)malloc(sizeof(int)*2);
                   	uint32_t *newDjD=(uint32_t*)malloc(sizeof(int)*2);

                   	newDjC[0]=DC[i];
                   	newDjC[1]=DC[k];
                   	newDjD[0]=DD[i];
                   	newDjD[1]=DD[k];

                   	FindIntermediateColl (r, DX0[i], Dsteps[i],
                                    	DX0[k], Dsteps[k], newDjC, newDjD);

                   	DC[i]=newDjC[0];
                   	DC[k]=newDjC[1];
                   	DD[i]=newDjD[0];
                   	DD[k]=newDjD[1];

                   	free(newDjC);
                   	free(newDjD);
   			}
   			 
   		 	if (DC[i]!=DC[k]){
   				for (int h = 0;h<nCollisFinal+nCollisTot+1;h++){
                   		if((DC[i]==scA[h] && DC[k]==scB[h]) || (DC[i]==scB[h] && DC[k]==scA[h])){
                           		 b++;
                           		 break;}}
                   		if (b==0){ //if the Collsion is new
	   				if (DC[i]<DC[k]){
	   					 tempA[nCollisIrT]=DC[i];
	   					 tempB[nCollisIrT]=DC[k];}
	   				 else{
	   					 tempA[nCollisIrT]=DC[k];
	   					 tempB[nCollisIrT]=DC[i];}
	   				 //printf("new interrank %d and %d Collis bw %d and %d on %d \n", i, k, tempA[nCollisIrT], tempB[nCollisIrT], DD[i]);
   				 	nCollisIrT++;
				}//if b==0
   			 } //if DCi!=DCk
		} // if a==0
    }} // for i for k

    //eliminates duplicates between new collisions
    if (nCollisIrT>0){
   	scA[nCollisFinal+nCollisTot]=tempA[0];
        scB[nCollisFinal+nCollisTot]=tempB[0];
        printf("first interrank collis bw %d and %d \n", scA[nCollisFinal+nCollisTot], scB[nCollisFinal+nCollisTot]);

        for (int i = 1; i < nCollisIrT; i++){
        	int a=0;
                for (int k = 0; k<i; k++){
                if(tempA[i]==tempA[k] && tempB[i]==tempB[k]){
                        a++;
                        break;}}
                if (a==0){
                        scA[nCollisFinal+nCollisTot+nCollisIr]=tempA[i];
                        scB[nCollisFinal+nCollisTot+nCollisIr]=tempB[i];
                        printf("interrank collis bw %d and %d \n", scA[nCollisFinal+nCollisTot+nCollisIr], scB[nCollisFinal+nCollisTot+nCollisIr]);
                        nCollisIr++;}
            	}
    }

    //frees on device
    free(tempA);
    free(tempB);

    nCollisTot=nCollisTot+nCollisIr;
    printf("nTot of this iteration %d \n", nCollisTot);
    } //ROOT

    MPI_Barrier(MPI_COMM_WORLD);
    
    //each process is updated on the number of collisions found in this iteration
    MPI_Bcast(&nCollisTot,1, MPI_INT,ROOT,MPI_COMM_WORLD);
    
    free(DX0);
    free(DD);
    free(Dsteps);
    free(DC);
    
    free(CA);
    free(CB);

    //counts number of iterations in a row that didn't give any result
    if (nCollisTot==0) nMoot++;
    else nMoot=0;

    //stops the program after a certain number of moot iterations 
    if (nMoot>=ITERMAX){
   	MPI_Bcast(&nMoot,1, MPI_INT,ROOT,MPI_COMM_WORLD);
   	MPI_Barrier(MPI_COMM_WORLD);
   	if (rank ==ROOT) printf("no new Collision have been found for %d iterations, the required number of Collisions migth be too high \n", nMoot);
   	break;}
    
    nCollisFinal= nCollisFinal+nCollisTot;
    MPI_Bcast(&nCollisFinal,1, MPI_INT,ROOT,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==ROOT) printf("nFin %d \n", nCollisFinal);
    nIter++;

    /*if (nCovered>=PRIME){
            	//MPI_Bcast(&nCovered,1, MPI_INT,ROOT,MPI_COMM_WORLD);
            	MPI_Barrier(MPI_COMM_WORLD);
            	if (rank==ROOT) printf(" %d points searched, the required number of Collisions migth be too high \n", nCovered);
            	break;
    }*/

}//WHILE

//for last iteration
MPI_Barrier(MPI_COMM_WORLD);
MPI_Bcast(&nCollisFinal,1, MPI_INT,ROOT,MPI_COMM_WORLD);
if (rank==ROOT) {
    printf("nDef reached %d \n", nCollisFinal);
    for (int i=0; i<nCollisFinal; i++) printf ("%d and %d \n", scA[i], scB[i]);
}

MPI_Barrier(MPI_COMM_WORLD);
TIMER_STOP;

if (rank==ROOT) {
    printf("in %d iterations %d points in the set have been searched \n", nIter, nCovered);
    printf("running time: %f microseconds\n",TIMER_ELAPSED);

    //save in csv subroutine
    printf("Do you want to save the ouput in a csv? (0=no/1=yes) \n");

    int answ;
    scanf("%d", &answ);
    if(answ==1){
   	 FILE *fp;
   	 char filename[100];
   	 printf("Type file name \n ");
   	 //gets(filename);
   	 scanf("%99s", filename);
   	 strcat(filename,".csv");
   	 fp=fopen(filename,"w+");
   	 for(int i = 0; i<nCollisFinal; i++){
   		 fprintf(fp,"\n %d,%d", scA[i], scB[i]);
   	 }
   	 fclose(fp);
    }    
}

//frees on device
free(scA);
free(scB);
free(x0glob);

MPI_Finalize();
return 0;
}
