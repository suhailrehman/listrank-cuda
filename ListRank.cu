/*
 *	List Ranking using GPU
 *	(Hellman&JaJa Algorithm)
 *      Suhail Rehman
 *	MS by Research, CS
 *	IIIT-Hyderabad
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
//#include <curses.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <ListRank_kernel.cu>

// includes, auxillary functions
#include <auxfunctions.h>

#define MAX 100

#define CPULIMIT 100 

//#define K(size) (int)(2*ceil(pow((log(size)/log(2)),2)))

#define K(size) ceil(log(size)/log(2)) 

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

//Major Functions

void ListRankOnDevice(int *, int *,int *,Sublist *,int*, int,int, int, int, int);
void ListRankSequential(int*,int*,int);
void SublistListRankOnDevice(Sublist *, int,int, int,int,int);
Sublist *S;


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(const int argc,const char** argv) {

	int *VAL,*VAL_DEVICE, *VAL_GPU_RESULT;
	int *SUC,*SUC_DEVICE;
	int size=MAX, seed;
	
	srand(2012);
	unsigned int hTimer;
	double gpuTime, cpuTime;
	char *file, *csv;

	CUT_DEVICE_INIT(argc,argv);
    	CUT_SAFE_CALL(cutCreateTimer(&hTimer));

	//User Info to be Printed Onscreen
	printf("\nList Ranking using GPU\n");
	printf("========================\n\n");		
	printf("This program Ranks every element of a specified List\n");
	printf("Try `%s --help' for more information.\n",argv[0]);
	
	
		
	if(cutCheckCmdLineFlag( argc, argv, "help"))
	{
		printf("\nUsage:\n");
		printf("%s [arguments] \n",argv[0]);
		printf("\nArguments:\n\n");
		printf("--file=<filename>    : Open File named <filename>\n");
		printf("                       Text file of a number in each line, first number is list size\n");
		printf("                       The rest of the numbers should be contents of the list's\n");
		printf("                       sucessor array\n");
		printf("--size=<value>       : Generates a Random list of size <value>\n");
		printf("--seed=<value>       : Use seed the Randomizer with <value>\n");
		printf("--print              : Print out the result lists (CPU & GPU)\n");
		printf("--noverify           : Do not verify the results\n");
		printf("--csv=<filename>     : Append timing results in <filename> as a CSV\n");
		exit(1);
	}
		
	//read from file argument
					
	else if( cutGetCmdLineArgumentstr(argc, argv, "file", &file) )
	{
		printf("\nReading List from File \"%s\"...",file);
		//Read List from File
      		SUC=ReadList(file,&size);
		
	}
	
	//use random generation

	else if( cutGetCmdLineArgumenti( argc, argv, "size", &size) )
	{
		cutGetCmdLineArgumenti( argc, argv, "seed", &seed);
		printf("\nGenerating ");
		fflush(stdout);
		//Old Generation Function (Slow!)
		//SUC=generate_list(size,seed);
		
		//New Generation Function (from Bader)
		if(cutCheckCmdLineFlag( argc, argv, "seq"))
		{
			printf("Ordered List...");
			SUC=load_data_sequential(size);
		}
		else
		{
			printf("Random List...");
			SUC=load_data_random(size,seed);
		}


		printf("Complete.");
		fflush(stdout);

		printf("\nSize of Sublist: %d",sizeof(Sublist));

		if( cutGetCmdLineArgumentstr(argc, argv, "writelist", &file) )
		{
			printf("\nWriting to List Data to File...\n");
			writelist(file,SUC,size);
		}
			
	}	

	//Exit if not enough data to proceed			
	else		
	{

		printf("\nError: Not enough arguments to proceed, try --help for instructions.\n\n");
		exit(1);	
	}	
	
	VAL=(int*)malloc(size*sizeof(int));
	init2zero(VAL,size);

    	printf("\nInput List Size: %d\n", size);
	
	int sublist_size=(int)ceil(K(size));//(int)ceil(K*(log(size)/log(2)));
	int num_threads=(int)ceil((float)size/(float)sublist_size);
	
	printf("\nNumThreads: %d",num_threads);
	printf("\nSublist Size: %d",sublist_size);
	Sublist *S_DEVICE;


	//Compute Depth of Recursion required
	
	int no_of_levels=1;
	int temp=size;

	
	while(temp>CPULIMIT)
	{
		temp=(int)ceil((float)temp/K(size));//(float)ceil(K*(log(temp)/log(2))));
		printf("\nLevel %d : List Size: %d",no_of_levels, temp);
		no_of_levels++;
	}

	printf("\nRecursion Depth=%d, Final Sublist Size= %d",no_of_levels,temp);

	//Allocate Memory Required for final CPU Reduce Sublist;
	S=(Sublist*)malloc(num_threads*sizeof(Sublist));

	//Allocate Device Versions of the two arrays
	
	cudaMalloc((void**)&VAL_DEVICE,size*sizeof(int));
	cudaMalloc((void**)&SUC_DEVICE,size*sizeof(int));

	//for(i=0;i<no_of_levels;i++)
	cudaMalloc((void**)&S_DEVICE,no_of_levels*num_threads*sizeof(Sublist));
	



	cudaMemcpy(VAL_DEVICE,VAL,size*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(SUC_DEVICE,SUC,size*sizeof(int),cudaMemcpyHostToDevice);
	
	//Precompute Head and Tail before starting timer
	int tail;
	int head=ListHead(SUC,size,&tail);

	//Find Splitters and create Splitters array
	int *Splitters=(int*)malloc(num_threads*sizeof(int));
	int *Splitter_D;
	FindSplittersLB(SUC, Splitters, size, sublist_size);
	//FindSplittersRand(SUC, Splitters, size, sublist_size);
    	cudaMalloc((void**)&Splitter_D,num_threads*sizeof(int));
	
	cudaMemcpy(Splitter_D,Splitters,num_threads*sizeof(int),cudaMemcpyHostToDevice);
	//Start Timer for GPU
	
	printf("\nEntering GPU"); fflush(stdout);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
 	CUT_SAFE_CALL( cutResetTimer(hTimer) );
    	CUT_SAFE_CALL( cutStartTimer(hTimer) );	

	//Call Wrapper Function for GPU Kernel

	ListRankOnDevice(VAL_DEVICE, SUC_DEVICE,SUC,S_DEVICE,Splitter_D,num_threads,sublist_size,size,head,tail);

    	CUDA_SAFE_CALL( cudaThreadSynchronize() );
    	CUT_SAFE_CALL(cutStopTimer(hTimer));
    	gpuTime = cutGetTimerValue(hTimer);
	
	//Copy VAL array (containing ranks) to Host
	VAL_GPU_RESULT=(int*)malloc(size*sizeof(int));
        cudaMemcpy(VAL_GPU_RESULT,VAL_DEVICE,size*sizeof(int),cudaMemcpyDeviceToHost);

        // Free device arrays
        cudaFree(VAL_DEVICE); VAL_DEVICE=NULL;
        cudaFree(SUC_DEVICE); SUC_DEVICE=NULL;

	//Report GPU List Ranking Time
	printf("\nGPU Ranking time : %.4f msec, %.3f K-Elements/sec",gpuTime,(double(size)/gpuTime));
    
	//Reset & Start Timer for CPU
    	CUDA_SAFE_CALL( cudaThreadSynchronize() );
    	CUT_SAFE_CALL( cutResetTimer(hTimer) );
    	CUT_SAFE_CALL( cutStartTimer(hTimer) );
    
	//Compute List Rank on the CPU for comparison
        ListRankSequential(VAL,SUC,size);	 
    
	//Stop Timer and store in variable cpuTime
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
    	CUT_SAFE_CALL(cutStopTimer(hTimer));
    	cpuTime = cutGetTimerValue(hTimer);

	//Report Sequential List Ranking Time and Speedup of GPU vs. CPU
    	printf("\nCPU Ranking time : %.4f msec, %.3f K-Elements/sec\n", cpuTime,(double(size)/cpuTime));
	printf("Raw Speedup: %f\n",cpuTime/gpuTime);

	
	if( cutGetCmdLineArgumentstr(argc, argv, "csv", &csv) )
		writecsv(csv,size,seed,cpuTime,gpuTime);

	if(cutCheckCmdLineFlag( argc, argv, "print"))
	{
		printf("\nCPU Result:\n");
		DisplayList(VAL,SUC,size);
		printf("\n\nGPU Result:\n");
		DisplayList(VAL_GPU_RESULT,SUC,size);
		printf("\n");
	}

    	
	//Compare CPU and GPU results
        if(!cutCheckCmdLineFlag( argc, argv, "noverify"))
	{
		bool res=verify(VAL,VAL_GPU_RESULT,size);
		printf("\nCPU vs. GPU result equivalence test: ");
		printf("%s\n", (res==true) ? "PASSED" : "FAILED");
	}

	//Free dynamic variable on Host
	free(VAL_GPU_RESULT);
}


////////////////////////////////////////////////////////////////////////////////
//// CUDA Entry Wrapper Function for List Ranking on GPU
//// Creates sublist and passes it on to SublistRankOnDevice for recursive
//// List Ranking
////////////////////////////////////////////////////////////////////////////////
void ListRankOnDevice(int *VAL, int *SUC, int *SUC_H,Sublist *S_DEVICE,int* Splitters, int num_threads,int sublist_size, int size, int head, int tail)
{    

	// Setup the execution configuration    
	// Single Block, 1 Thread per List element
	int y;
	int blocksize=num_threads>512?512:num_threads;
	int gridsize=(int)ceil(float(num_threads)/512.0);
	dim3 dimBlock(blocksize,1,1);
	dim3 dimGrid(gridsize,1,1);

	Kernel1LB<<<dimGrid,dimBlock>>>(VAL,SUC,S_DEVICE,sublist_size,size,head,tail,Splitters,num_threads);

	//Kernel1<<<dimGrid,dimBlock>>>(VAL,SUC,S_DEVICE,sublist_size,size,head,tail);


	Kernel2<<<dimGrid,dimBlock>>>(VAL,SUC,S_DEVICE,sublist_size,size,head,tail);


	if(num_threads <=CPULIMIT)
	{

		cudaMemcpy(S,S_DEVICE,num_threads*sizeof(Sublist),cudaMemcpyDeviceToHost);

		CPUSublistRank(S,num_threads);

		cudaMemcpy(S_DEVICE,S,num_threads*sizeof(Sublist),cudaMemcpyHostToDevice);
	}


	else
	{
		int stride=num_threads;
		int size2=num_threads;
		int sublist_size2=(int)K(size);//(int)ceil(K*(log(size2)/log(2)));
		int num_threads2=(int)ceil((float)size2/(float)sublist_size2);
		int level=1;

		SublistListRankOnDevice(S_DEVICE,num_threads2,sublist_size2,size2,level,stride);
	}


	blocksize=size>512?512:size;
	gridsize=(int)ceil(float(size)/512.0);
	if(gridsize>65535)
	{
		gridsize=65535;

	//	printf("\nGrid Size: %d",gridsize);
		
		y=(int)ceil((float)size/(65535.0*512.0));
	}
	else
	{
		y=1;
	}		
	
	
	dim3 dimGrid1(gridsize,y,1);
	dim3 dimBlock1(blocksize,1,1);


	Kernel4<<<dimGrid1,dimBlock1>>>(VAL,SUC,S_DEVICE,sublist_size,size,head);
}

////////////////////////////////////////////////////////////////////////////////
// Recursive CUDA Wrapper Function for Ranking Sublists on GPU
// Function recurses until sublist is less than CPULIMIT
////////////////////////////////////////////////////////////////////////////////
void SublistListRankOnDevice(Sublist *S_DEVICE, int num_threads,int sublist_size, int size, int level, int stride)
{    

	// Setup the execution configuration    
	int blocksize=num_threads>64?64:num_threads;
	int gridsize=(int)ceil(float(num_threads)/64.0);

	dim3 dimBlock(blocksize,1,1);
	dim3 dimGrid(gridsize,1,1);

	int tail;
	int head=0;


	//Find GPU Tail
	int *d_tail;
	int blocksize1=size>64?64:size;
	int gridsize1=(int)ceil(float(size)/64.0);

	dim3 dimBlock1(blocksize1,1,1);
	dim3 dimGrid1(gridsize1,1,1);




	CUDA_SAFE_CALL( cudaMalloc((void**) &d_tail, sizeof(int)) );
	findsublisttail<<< dimGrid1, dimBlock1>>>(S_DEVICE+((level-1)*stride), d_tail,size);
	CUDA_SAFE_CALL( cudaMemcpy( &tail, d_tail, sizeof(int), cudaMemcpyDeviceToHost) );


	SublistKernel1<<<dimGrid,dimBlock>>>(S_DEVICE+((level-1)*stride),S_DEVICE+(level*stride),sublist_size,size,head,tail,level);


	SublistKernel25<<<dimGrid,dimBlock>>>(S_DEVICE+((level-1)*stride),S_DEVICE+(level*stride),sublist_size,size,head,tail);

	if(num_threads<=CPULIMIT)
	{



		/*
		//    	printf("\nEntered CPU Reduce Phase, Sublist Size= %d",num_threads);
		//	fflush(stdout);
		*/
		cudaMemcpy(S,S_DEVICE+(level*stride),num_threads*sizeof(Sublist),cudaMemcpyDeviceToHost);

		CPUSublistRank(S,num_threads);

		cudaMemcpy(S_DEVICE+(level*stride),S,num_threads*sizeof(Sublist),cudaMemcpyHostToDevice);
		


	//	SublistKernel3<<<dimGrid,dimBlock>>>(S_DEVICE+(level*stride),num_threads,0);

	}

	else
	{

		//Recurse and Rank Sublist on GPU

		int size2=num_threads;
		int sublist_size2=(int)ceil(log(size2)/log(2));
		int num_threads2=(int)ceil((float)size2/(float)sublist_size2);

		SublistListRankOnDevice(S_DEVICE,num_threads2,sublist_size2,size2,level+1,stride);
	}


	SublistKernel4<<<dimGrid1,dimBlock1>>>(S_DEVICE+((level-1)*stride),S_DEVICE+(level*stride),sublist_size,size,head);

}


////////////////////////////////////////////////////////////////////////////////
//// Sequential List Ranking Function
////////////////////////////////////////////////////////////////////////////////
void ListRankSequential(int *VAL, int *SUC, int size)
{
        int i;
        long long int Lsum=0;

	/*
        //Caluclate the sum of the successor array
        for(i=0;i<size;i++)
	{
		if(SUC[i]==-1) continue;
		Zsum+=SUC[i];
        }
	
        //Set the start position of Head of List
        i=abs(int(long(long(size)*(long(size)-1)/2)-Zsum));
        */
	int tail;
	
	i=ListHead(SUC, size, &tail);
	//printf("\nHead of List: %d\n",i);
        
        
        //Traverse and Update the VAL array with List Rank
        while(SUC[i]!=-1)
        {
       		 VAL[i]=Lsum++;
                 i=SUC[i];
	}
        
	VAL[i]=Lsum;
	//printf("\nSeq. List Rank Final Count: %d",Lsum);
}










