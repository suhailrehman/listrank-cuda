//Function Prototypes
//For List Ranking
int * ReadList(char* ,int *);
void writecsv(char *,int,int,float,float);
void DisplayList(int *,int *,int);
void init2zero(int *,int);
int *generate_list(int, int);
int *generate_list1(int, int);
bool verify(int *,int *,int );
int ListHead(int *, int , int *);
void getNumBlocksAndThreads(int, int, int, int*, int*);
int GPUlistHead(int *, int,int *);
void CPUSublistRank(Sublist *,int);
void writelist(char*,int *,int);
void sublistverify(Sublist *, Sublist *, int);


//Function Declarations

///////////////////////////////////////////////////////
//Function ListHead
//Function to Find Head and Tail from Sucessor Array
//Return Value: Head (Int) and Tail passed as parameter
///////////////////////////////////////////////////////
int ListHead(int *SUC, int size, int *tail)
{
        int i;
	long long int temp;
        long long int Zsum=0;

        //Caluclate the sum of the successor array
        for(i=0;i<size;i++)
	{
		if(SUC[i]==-1)
		{ 
			*tail=i;
			continue;
		}
		Zsum+=SUC[i];
        }
	
        //Set the start position of Head of List
    	temp=size;
	temp*=(temp-1);
	

	temp=temp/2;
	 
        
	i=(int)abs(temp-Zsum);

        return i; 
        
       
}


//////////////////////////////////////////////////////
// Function ReadList
// Function to Read a List from File
// Return Value: Integer Pointer to Succesor Array
//               Size of List passed as parameter
/////////////////////////////////////////////////////
int * ReadList(char *filename, int *size)
{

        FILE *fp;
        int i=0;
        fp=fopen(filename,"r");

        fscanf(fp,"%d",size);

	if(fp==NULL)
	{
		printf("\nError: Cannot open specified file, exiting...\n");
		exit(1);
	}

	int *SUC=(int*)malloc(*size*sizeof(int));
        
	for(i=0;i<(*size);i++)
        {
                fscanf(fp,"%d",&(SUC[i]));
        }

        fclose(fp);
	
	return SUC;
}

//////////////////////////////////////////////////////
// Function DisplayList
// Function to Read a List from File
/////////////////////////////////////////////////////

void DisplayList(int VAL[], int SUC[], int size)
{
        int i;

        for(i=0;i<size;i++)
        {
                printf("\n%d:%d(%d) ",i,VAL[i],SUC[i]);
        }

}

//Initializes Array A to 0
void init2zero(int *A,int size)
{
        int i;
        for(i=0;i<size;i++) A[i]=0;
}


/////////////////////////////////////////////////////
// Function generate_list
// Function to Generate a Random Integer List 
// given Size and Seed Value
// Return Value: Integer Pointer to Successor Array
/////////////////////////////////////////////////////

int *generate_list(int size, int seed)
{
	
	int *list=(int*)malloc(size*sizeof(int));
	int val,remain=size,i,j,head,count=0;
	int *array=(int*)malloc(size*sizeof(int));


	for(i=0;i<size;i++)
	{
		 array[i]=i;

	}
	srand(seed);

	head=rand()%size;
	val=i=head;

	

	while(remain>1)
	{

		remain--;
		
		for(j=val;j<remain;j++) array[j]=array[j+1];
		val=rand()%remain;
		list[i]=array[val];
		i=list[i];
		count++;
	}	
	
	list[array[0]]=-1;

	return list;
}

//////////////////////////////////////////////////////
// Function verify
// Function to Compare two Intger Pointers to Lists
// Return Value: true on success
/////////////////////////////////////////////////////

bool verify(int *list1, int *list2, int size)
{
	
	int count=0,count2=0;

	for(int i=0;i<size;i++)
	{
		if(list1[i]!=list2[i]) 
		{
			//	printf("%d :: %d : %d\n",i, list1[i], list2[i]);
			count2++;
		
		}
		
		else if(list1[i]==0) count++;
	}
	
	if(count>1)
		printf("\nWarning, %d Zero Elements in Ranking",count);	
	
	if(count2>0)
	{
		printf("\n%d Variations",count2);	
		return false;
	}
	return true;
}


//////////////////////////////////////////////////////////
// Function writecsv 
// Function to append timing data to a file in CSV format
//////////////////////////////////////////////////////////
void writecsv(char *filename, int size, int seed, float cputime, float gputime)
{
	
	FILE *fp;
        
        fp=fopen(filename,"a");

        fprintf(fp,"\n%d,%d,%.4f,%.4f",size,seed,cputime,gputime);
	
	fclose(fp);
}


void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int *blocks, int *threads)
{
       if (n == 1)
            *threads = 1;
       else
            *threads = (n < maxThreads*2) ? n / 2 : maxThreads;
        
       *blocks = n / (*threads * 2);

       *blocks = min(maxBlocks, *blocks);
    
}

//////////////////////////////////////////////////////////
// Function GPUlistHead 
// Function to find the Head & Tail elements of a list
// using GPU Reduction
// Return Values: Head of List, Tail as a parameter
//////////////////////////////////////////////////////////
int GPUlistHead(int *d_idata, int size,int *tail)
{
	int blocksize, gridsize, blocksize1, gridsize1;
	long long int gpu_result=0, temp=0;

	int *d_odata, *h_odata, *d_tail;

	getNumBlocksAndThreads(size, 256, 256, &gridsize, &blocksize);
	
	h_odata=(int*)malloc(gridsize*sizeof(int));

	//CUDA_SAFE_CALL( cudaMalloc((void**) &d_idata, size*sizeof(int)) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_odata, gridsize*sizeof(int)) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_tail, sizeof(int)) );
	
	//CUDA_SAFE_CALL( cudaMemcpy(d_idata, SUC, size*sizeof(int), cudaMemcpyHostToDevice) );
	//CUDA_SAFE_CALL( cudaMemcpy(d_i2data, SUC, size*sizeof(int), cudaMemcpyHostToDevice) );
    	//CUDA_SAFE_CALL( cudaMemcpy(d_odata, d_idata,  gridsize*sizeof(int), cudaMemcpyDeviceToDevice) );
	//printf("\nGPU Reduce: Gridsize: %d. Blocksize %d",gridsize,blocksize);	
	dim3 dimBlock(blocksize,1,1);
    	dim3 dimGrid(gridsize,1,1);

	int smemSize = blocksize * sizeof(int);

	//GPU Reduce
	reduce6<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata,size);
	cudaThreadSynchronize();	
	CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_odata, gridsize* sizeof(int), cudaMemcpyDeviceToHost) );


	//CPU reduce the remaining values
	for(int i=0; i < gridsize; i++)
	{
	    gpu_result += h_odata[i];
	}

	gpu_result++;
//	printf("GPU Reduce Result:%lu",gpu_result);
//	fflush(stdout);
	temp=size;
	temp*=(temp-1);
	
	temp=temp/2;
	 
	
        
	gpu_result=(int)abs(temp-gpu_result);
     


	
	//GPU Find Tail
	
	blocksize1=size>64?64:size;
    	gridsize1=(int)ceil(float(size)/64.0);
        
	dim3 dimBlock1(blocksize1,1,1);
        dim3 dimGrid1(gridsize1,1,1);
	
	findtail<<< dimGrid1, dimBlock1>>>(d_idata, d_tail,size);



	CUDA_SAFE_CALL( cudaMemcpy( tail, d_tail, sizeof(int), cudaMemcpyDeviceToHost) );

	
	//printf("\nGPU Head of List: %d",gpu_result);
	//printf("\nGPU Tail of List: %d\n", *tail);
	return (int) gpu_result;

}



void CPUSublistRank(Sublist *S,int size)
{

	int prefix=S[0].prefix;
	int p=S[0].succ;
	int count=0;

	while(p!=-1)
	{

		S[p].prefix=S[p].prefix+prefix;
		prefix=S[p].prefix;
//		printf("\nCPURank: %d:%d - suc %d",p, S[p].prefix,S[p].succ);             
		p=S[p].succ;
		count++;
	                          
	}
	
//	printf("\nCPU Ranked %d elements",count);
//	printf("\nFinal CPU Rank Tail: %d",temp);
}


//////////////////////////////////////////////////////////
// Function writelist 
// Function to write the current processed list to file
//////////////////////////////////////////////////////////
void writelist(char* filename,int *SUC,int size)
{

	FILE *fp;
	int i;
	fp=fopen(filename,"w");

	fprintf(fp,"%d\n",size);

	if(fp==NULL)
	{
		printf("\nError: Cannot open specified file, exiting...\n");
		exit(1);
	}


	for(i=0;i<size;i++)
	{
		fprintf(fp,"%d\n",SUC[i]);
	}

	fclose(fp);

}

//////////////////////////////////////////////////////////
// Function sublistverify
// Function to verify the contents of two sublists
//////////////////////////////////////////////////////////

void sublistverify(Sublist *list1, Sublist *list2, int size)
{
	
	int count=0,count2=0;

	printf("\n");
	for(int i=0;i<size;i++)
	{

		printf("%3d ",list2[i].prefix);
		if(list1[i].prefix!=list2[i].prefix) 
		{
		       //printf("\n%d :: %d : %d",i, list1[i].prefix, list2[i].prefix);
			count2++;
		
		}
		
		else if(list1[i].prefix==0) count++;
	}


	printf("\n");
	for(int i=0;i<size;i++)
		printf("%3d ",list2[i].succ);
	
	if(count>1)
		printf("\nWarning, %d Zero Elements in Ranking",count);	
	
	if(count2>0)
	{
		printf("\n%d Variations",count2);	
///		return false;
	}
//	return true;
}

//Bader's List Generation Function
//From Cell List Ranking Code
int * load_data_random(unsigned long array_size, int seed) {
 
  array_size--;
  unsigned long i;
  int i1,j1,temp;
  int current=0;
  int *data1;
  int *data=(int*)malloc(array_size*sizeof(int));
  data1 = (int *) malloc(127 + (array_size+1)*sizeof(int));
  while (((int) data1) & 0x7f) ++data1;

  srand48(seed);
  //  srand48(time(NULL));

  for (i=0; i<array_size; i++) data1[i] = i+1;
  data1[array_size] = -1;

  //n swaps for fairly random access
  for(i=0;i<2*array_size;i++) {
    i1 = j1 = (int)(drand48() * (array_size-1));
    while(i1 == j1)
      j1 = (int)(drand48() * (array_size-1));//rand() % (array_size-1);
    
    temp = data1[i1];
    data1[i1] = data1[j1];
    data1[j1] = temp;
  }
  
  //make random permutation
  data[0] = data1[0];
  current = data[0];
  for(i=0;i<array_size;i++) {
    data[current] = data1[i+1];
    current = data1[i+1];
  }
  data[array_size] = -1;

  return data;
}

int * load_data_sequential(unsigned long array_size) 
{

	int *data=(int*)malloc(array_size*sizeof(int));
	unsigned long i;

	for (i=0; i<array_size; i++) data[i] = i+1;
	data[array_size-1] = -1; // data[array_size] acts as a ground for the end os list.

	return data;
}

void FindSplittersLB(int *SUC, int *Splitters, int size, int splitter_size)
{
	int i,j=0,k=0;


	int tail;

	i=ListHead(SUC, size, &tail);
	printf("\nHead of List: %d",i);
	
	while(SUC[i]!=-1)
	{
		if(k%splitter_size==0)
		{
			Splitters[j++]=i;
			//printf("\n%d: %d",j-1,Splitters[j-1]);
		}


		i=SUC[i];
		k++;
	}

}

void FindSplittersRand(int *SUC, int *Splitters, int size, int splitter_size)
{
	int i,j=0,k=0;


	int tail,split;

	i=ListHead(SUC, size, &tail);
	printf("\nHead of List: %d",i);
	Splitters[j++]=i;
	int *check=(int*)malloc(size*sizeof(int));
	
	for(k=0;k<size;k++) check[k]=0;
	
	int lim= (int)ceil((float)size/(float)splitter_size);
	for(k=1;k<lim;k++)
	{
		split=rand()%size;
		while(check[split]!=0) split=rand()%size;
		Splitters[k]=split;
		check[split]=1;
	}

}
