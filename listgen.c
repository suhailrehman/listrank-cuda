#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int *generate_list(int size, int seed);
void print_list(int *list,int size);

int main(int argc, char *argv[])
{
	int size,seed,*list;	

	if(argc==3)
	{
		size=atoi(argv[1]);
		seed=atoi(argv[2]);
	}
	
	else
	{
		printf("Error- specify two arguments\n");
		exit(1);
	}
	
	
	list=generate_list(size,seed);
	print_list(list,size);

}

int *generate_list(int size, int seed)
{
	printf("In\n");	
	int *list=(int*)malloc(size*sizeof(int));
	int val,remain=size,i,j,head,count=0;
	int *array=(int*)malloc(size*sizeof(int));

	for(i=0;i<size;i++)
	{
		 array[i]=i;
//		 printf("A[%d]=%d ",i,array[i]);
	}
	srand(seed);

	//Find Head Element
	head=rand()%size;
	val=i=head;
	printf("Head=%d, Size=%d\n",i,remain);
	

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
	printf("Count=%d\n",count);
	return list;
}
	
void print_list(int *list, int size)
{
	int i;
	printf("\n");	
	for(i=0;i<size;i++)
	{
		printf("%d ",list[i]);
	}
	
	printf("\n");
}

int *generate_list(int size, int seed)
{
	printf("In\n");	
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
	printf("Head=%d, Size=%d\n",i,remain);
	

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
	printf("Count=%d\n",count);
	return list;
}
