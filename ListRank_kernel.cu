#ifndef _LISTRANK_KERNEL_H_
#define _LISTRANK_KERNEL_H_


typedef struct {
    int head;
    int scratch;
    int prefix;
    int succ;
    int succ2;
} Sublist;



__global__ void Kernel1(int *VAL, int *SUC, Sublist *S,int s_size, int size, int head,int tail)
{
	
	int index=(blockIdx.x*blockDim.x)+threadIdx.x;
	int split;

	//Set the Splitter - first element of block of logn elements
	
	//// printf("\n%d: ",index);
	//fflush(stdout);

	split=index*s_size;
	if(split==tail||split==head) split++;
	if(index==0) split=head;
	
	
	if(split>=size) return;	
	//// printf("Index=%d\n",split);
	
	S[index].head=split;
	S[index].scratch=SUC[split];
	SUC[split]=-(index);
	//printf("I: %d, Splitter=%d, SUC[i]=%d, Sublist.head=%d, Sublist.scratch=%d",index, split, SUC[split], S[index].head, S[index].scratch);
	

}

__global__ void Kernel1LB(int *VAL, int *SUC, Sublist *S,int s_size, int size, int head,int tail, int *Splitters, int numsplitters)
{
	
	int index=(blockIdx.x*blockDim.x)+threadIdx.x;

	int split;

	//Set the Splitter - first element of block of logn elements
	
	//printf("\n%d: ",index);
	//fflush(stdout);
	if(index>=numsplitters) return;
	split=Splitters[index];

	
	if(split>=size) return;	
	//// printf("Index=%d\n",split);
	
	S[index].head=split;
	S[index].scratch=SUC[split];
	SUC[split]=-(index);
	//printf("I: %d, Splitter=%d, SUC[i]=%d, Sublist.head=%d, Sublist.scratch=%d",index, split, SUC[split], S[index].head, S[index].scratch);
	

}

__global__ void Kernel2(int *VAL, int *SUC, Sublist *S,int s_size, int size, int head, int tail)
{

	int index=(blockIdx.x*blockDim.x)+threadIdx.x;

	//int split=index*sublist_size;

	if(index*s_size>=size) return;

	int p=S[index].scratch;
	int prefix=0;
	int temp=p;	
	int count=0;
	//Traverse and set sublist prefix and sucessor to point to sublist index
	while(p>=0)
	{
		
		//// printf("\nLoop1,%d:%d",index,p);
		//fflush(stdout);	
	

		temp=p;
		p=SUC[p];
		SUC[temp]=-(index);
		VAL[temp]=++prefix;
		count++;
	}
#ifdef __DEVICE_EMULATION__
	printf("\n%d:%d",index,count);
#endif
	if(temp==tail)
	{
		 VAL[temp]=prefix;
		  S[index].succ=-1;
		  S[index].succ2=1;
		  // // printf("\nLevel 0 Tail: %d, prefix: %d",index,prefix);
	}

	else if(p<0)
	{
		//Store the next sublist index
		S[index].succ=-p;
		S[index].succ2=-p;
		// printf("%d ",S[index].succ);
		S[-p].prefix=prefix;
		//// printf(" Prefix %d",S[-p].prefix);
		
	}

	if(index==0) S[0].prefix=0;
}


__global__ void Kernel3(int *VAL, int *SUC, Sublist *S,int s_size, int size, int head)
{


	int index=(blockIdx.x*blockDim.x)+threadIdx.x;



	if(index==0) 
	{
		int prefix=0;
		int p=S[0].succ;
		
			
		while(p!=-1)
		{
			
			S[p].prefix=S[p].prefix+prefix;
			prefix=S[p].prefix;
			//// printf("\n%d:%d",p, S[p].prefix);		
			p=S[p].succ;
			
		}

		
	}


}

//No. of Threads increased for Kernel 4
__global__ void Kernel4(int *VAL, int *SUC, Sublist *S,int sublist_size, int size, int head)
{
	
	
	int block=(blockIdx.y*gridDim.x)+blockIdx.x;
	int index=block*blockDim.x+threadIdx.x;

	if(index>=size) return;

	int suc=abs(SUC[index]);

	VAL[index]=VAL[index]+S[suc].prefix;
}

__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n)
{
	int blockSize=64;

	extern __shared__ int sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;
	
	while (i < n) 
	{
		sdata[tid] += g_idata[i] + g_idata[i+blockSize]; 
		i += gridSize; 
	}
	
	__syncthreads();
	
	if (blockSize >= 512) 
	{ 
		if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); 
	}
	if (blockSize >= 256) 
	{ 
		if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); 
	}
	if (blockSize >= 128) 
	{ 
		if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); 
	}
	
	if (tid < 32) 
	{
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
	
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}	



__global__ void findtail(int *g_idata, int *g_odata, unsigned int n)
{
	
	int index=(blockIdx.x*blockDim.x)+threadIdx.x;

	if(index<n&&g_idata[index]==-1) 
	{
		*g_odata=index;
		
		//// printf("Tail Index=%d",*g_odata);
	}
}

__global__ void findsublisttail(Sublist *S, int *g_odata, unsigned int n)
{
	
	int index=(blockIdx.x*blockDim.x)+threadIdx.x;

	if(index<n&&S[index].succ==-1) 
	{
		*g_odata=index;
		
		//// printf("Tail Index=%d",*g_odata);
	}
}
/*
__global__ void ListRankKernel2(int *LIST, int size)
{

	int i=(blockIdx.x*256)+threadIdx.x;

	if(i<size)
	{

		int temp=0;

		int mask=0xFFFF;
	

        	while((LIST[i]>>16)!=-1  && ((LIST[LIST[i]>>16]>>16)!=-1))	
		{
		
	
			//VAL[i]=VAL[i]+VAL[SUC[i]];

			temp=LIST[i]&mask;
			temp+= LIST[(LIST[i]>>16)]&mask;

			//SUC[i]=SUC[SUC[i]];

			temp+=(LIST[(LIST[i]>>16)]>>16)<<16;
	
			atomicExch(&LIST[i],temp);

		}

	}		
	
}*/

__global__ void SublistKernel1(Sublist *S1, Sublist *S,int s_size, int size, int head,int tail,int level)
{
	
	int index=(blockIdx.x*blockDim.x)+threadIdx.x;
	int split;

	//Set the Splitter - first element of block of logn elements
	
	//// printf("\nSublist Size: %d, size: %d",s_size,size);
	//fflush(stdout);
	split=index*s_size;
	if(split==tail||split==head) split++;
	if(index==0) split=head;
	
	
	if(split>=size) return;

	//// printf("\n%d: ",index);
	//fflush(stdout);
	//// printf("Index=%d\n",split);
	
	S[index].head=split;
	//// printf("\nDone");
	//fflush(stdout);
	S[index].scratch=S1[split].succ;
	S1[split].succ2=S1[split].succ;
	S1[split].succ=-(index);
	// printf("\nI: %d, Splitter=%d, SUC[i]=%d, Sublist.head=%d, Sublist.scratch=%d",index, split, S1[split].succ, S[index].head, S[index].scratch);
	
	//fflush(stdout);

}


__global__ void SublistKernel2(Sublist *S1, Sublist *S,int s_size, int size, int head, int tail)
{

	int index=(blockIdx.x*blockDim.x)+threadIdx.x;
	int flag=0;
	//int split=index*sublist_size;

	if(index*s_size>=size) return;


	if(index==0)
	{
		
		// printf("\nPrefix Array in SK2 before compute:\n");
		for(int i=0;i<3;i++)
		{
			// printf("%d ",S1[i].prefix);
		}
	}


	int prefix=0;
	int p=S[index].scratch;
	if(S1[p].succ<0) flag=1;
	if(!flag)  prefix=S1[S[index].head].prefix;
	
	// printf("\nSK2: Index: %d, Head: %d, Prefix %d, Succ %d",index, S[index].head, prefix, S1[p].succ);
	int temp=p;	
	

	//Traverse and set sublist prefix and sucessor to point to sublist index
	while(p>=0&&S1[p].succ>=0)
	{
				
		// printf("\nLoop1,%d:%d",index,p);
		//fflush(stdout);	
			
		
		// printf(" old S1.succ= %d", S1[p].succ);
		temp=p;
		//p=SUC[p];
		p=S1[p].succ;
		//SUC[temp]=-(index);
		
		S1[temp].succ2=S1[temp].succ;
		S1[temp].succ=(-index);
		//VAL[temp]=++prefix;
		// printf(" old prefix: %d",S1[temp].prefix);
		S1[temp].prefix+=prefix;
		// printf(" index: %d, new prefix: %d",temp, S1[temp].prefix);
		prefix=S1[temp].prefix;

	}

	if(p==tail)
	{
		 //VAL[temp]=prefix;
		  S1[p].prefix+=prefix;
		  S1[p].succ2=S1[p].succ;
		  S1[p].succ=(-index);
		  S[index].succ=-1;
		  // printf("\nIndex %d: Last item prefix: %d",index,S1[p].prefix);
		  // printf("\nPrefix Array in SK2 before compute:\n");
		  
		  for(int i=0;i<3;i++)
		  {
			// printf("%d ",S[i].prefix);
		  }


	}

	else if(S1[p].succ<0)
	{
		//Store the next sublist index
		int successor=-(S1[p].succ);
		S[index].succ=successor;
		//// printf("\nNext Sublist for Index %d, succ=%d, flag=%d",index, S[index].succ,flag);
		if(!flag)
			S[successor].prefix=prefix;
		else S[successor].prefix=S[index].prefix;
		//// printf(" Prefix %d",S[p].prefix);
		
	}

	if(index==head)
	{
		S[head].prefix=S1[head].prefix;
		//// printf("\nHead Prefix: %d",S[head].prefix);
	}


}

__global__ void SublistKernel3(Sublist *S, int size, int head)
{


	int index=(blockIdx.x*blockDim.x)+threadIdx.x;



	if(index==0) 
	{
		int prefix=0;
		int p=S[0].succ;
		
			
		while(p!=-1)
		{
			
			S[p].prefix=S[p].prefix+prefix;
			prefix=S[p].prefix;
			//// printf("\n%d:%d",p, S[p].prefix);		
			p=S[p].succ;
			
		}

		
	}


}

__global__ void SublistKernel4(Sublist *S1, Sublist *S,int sublist_size, int size, int head)
{
	
	int index=(blockIdx.y*blockDim.x)+(blockIdx.x*blockDim.x)+threadIdx.x;
	
	if(index>=size) return;

	int suc=abs(S1[index].succ);

	//VAL[index]=VAL[index]+S[suc].prefix;
	S1[index].prefix+=S[suc].prefix;
	//Restore Old Successor Value
	S1[index].succ=S1[index].succ2;

	__syncthreads();

	if(index==0)
	{

		// printf("\nPrefix array after addition:\n");
		for(int i=0;i<size;i++)
		{
				// printf("%3d ",S1[i].prefix);
		}

		// printf("\n");
	//	for(int i=0;i<size;i++)
			// printf("%3d ",S1[i].succ);

	}
}

__global__ void SublistKernel25(Sublist *S1, Sublist *S,int s_size, int size, int head, int tail)
{

	int index=(blockIdx.x*blockDim.x)+threadIdx.x;
	//int flag=0;
	//int split=index*sublist_size;

	if(index*s_size>=size) return;


	/*if(index==0)
	{
		
		// printf("\nPrefix Array in SK2 before compute:\n");
		for(int i=0;i<3;i++)
		{
			// printf("%d ",S1[i].prefix);
		}
	}*/


	int p=S[index].scratch;
	int prefix=S1[S[index].head].prefix;
	// printf("\nStarting SK2 Sweep for Element %d, Prefix: %d",index,prefix);
	int temp=p;

	while(p>0)
	{
	
	
		//// printf(" S1[%d].prefix=%d, succ=%d",temp,prefix,S1[temp].succ);
		temp=p;				
		
		S1[temp].succ2=S1[temp].succ;
		p=S1[temp].succ;
	
		if(p>0)
		{
			S1[temp].succ= -index;
		
			prefix+=S1[temp].prefix;
		
			S1[temp].prefix=prefix;
		}
	}

	if(temp==tail)
	{
		S1[temp].prefix+=prefix;
		S1[temp].succ2=-1;
		S1[temp].succ=-index;
		S[index].succ=-1;
		// printf("\nSK2 S Tail %d", index);
	}



	else if(p<0)
	{
		S[index].succ=-p;
		S[-p].prefix=prefix;

		// printf("\nNext Sublist for Index %d, succ=%d, prefix: %d",index, S[index].succ, S[S[index].succ].prefix);
	
	}


	if(index==0)
	{
		S[0].prefix=0;
	}
}

/*
__global__ void Kernel2Ord(int *VAL, int *SUC, Sublist *S,int s_size, int size, int head, int tail)
{

	int index=(blockIdx.x*blockDim.x)+threadIdx.x;

	__shared__ int VALs[BLOCKDIM];
	__shared__ int SUCCs[BLOCKDIM];

	//int split=index*sublist_size;

	if(index*s_size>=size) return;

		
	
	int p=S[index].scratch;
	int prefix=0;
	int temp=p;	
	
	//Traverse and set sublist prefix and sucessor to point to sublist index
	while(p>=0)
	{
		
		//// printf("\nLoop1,%d:%d",index,p);
		//fflush(stdout);	
	

		temp=p;
		p=SUC[p];
		SUC[temp]=-(index);
		VAL[temp]=++prefix;
		
	}

	if(temp==tail)
	{
		 VAL[temp]=prefix;
		  S[index].succ=-1;
		  S[index].succ2=1;
		  // // printf("\nLevel 0 Tail: %d, prefix: %d",index,prefix);
	}

	else if(p<0)
	{
		//Store the next sublist index
		S[index].succ=-p;
		S[index].succ2=-p;
		// printf("%d ",S[index].succ);
		S[-p].prefix=prefix;
		//// printf(" Prefix %d",S[-p].prefix);
		
	}

	if(index==0) S[0].prefix=0;
}

*/

#endif // #ifndef _LISTRANK_KERNEL_H_
