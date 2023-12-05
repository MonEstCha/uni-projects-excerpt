/*
*  This file is part of Christian's OpenMP software lab 
*
*  Copyright (C) 2016 by Christian Terboven <terboven@itc.rwth-aachen.de>
*  Copyright (C) 2016 by Jonas Hahnfeld <hahnfeld@itc.rwth-aachen.de>
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program; if not, write to the Free Software
*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*  
*  Note: The file was modified by MonEstCa and her submission team in the 
*  course of a software lab on Parallel Computing. 
*
*/

//bash: module load clang; NTHREADS=64 OMP_PROC_BIND=spread numactl --membind=4-7 make run-small

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/time.h>

#include <iostream>
#include <algorithm>

#include <cstdlib>
#include <cstdio>

#include <cmath>
#include <ctime>
#include <cstring>
#include <omp.h>

#include <numa.h>

#include <limits>
#include <hbw_allocator.h>
#include <hbwmalloc.h>
#include <numa.h>

using namespace std;


/**
 * Function prints the passed array
 */
void printArray(int *Array, size_t size){
	for(size_t i = 0; i < size; i++){
		cout << Array[i] << " ";
	}
	cout << endl;
}

//Quicksort adapted from https://www.geeksforgeeks.org/quick-sort/

/**
 * Function swaps the data at the passed addresses
 */
void swap(int* a, int* b) 
{ 
    int t = *a; 
    *a = *b; 
    *b = t; 
} 
 
/**
 * Function partitions the array
 * @param arr, the array (chunk) to partition
 * @param low, the start index of the chunk
 * @param high, the end index of the chunk
 */
long partition (int *arr, long low, long high) 
{ 
    long pivot = arr[high];    // pivot 
    long i = (low - 1);   
    // partition the array using last element as pivot
    for (long j = low; j <= high- 1; j++) 
    { 
        //if current element is smaller than pivot, increment the low element
        //swap elements at i and j
        if (arr[j] <= pivot) 
        { 
            i++;    // increment index of smaller element 
            swap(&arr[i], &arr[j]); 
        } 
    } 
    swap(&arr[i + 1], &arr[high]); 
    return (i + 1); 
} 
   
/**
 * Function sorts the passed array using quicksort
 */
void quickSort(int *arr, long low, long high) 
{ 
    if (low < high) 
    { 
        //partition the array 
        long pivot = partition(arr, low, high); 
   
        //sort the sub arrays obeying divide and conquer principle 
        quickSort(arr, low, pivot - 1); 
        quickSort(arr, pivot + 1, high); 
    } 

} 

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, (struct timezone*)0);
    return ((double)tv.tv_sec + (double)tv.tv_usec / 1000000.0 );
}

/**
  * helper routine: check if array is sorted correctly
  */
bool isSorted(int ref[], int data[], const size_t size){
	std::sort(ref, ref + size);
	for (size_t idx = 0; idx < size; ++idx){
		if (ref[idx] != data[idx]) {
			return false;
		}
	}
	return true;
}

/**
 * Function searches the passed array applying binary search
 */
int binarySearch(int array[], int x, int low, int high) {
  //cout << high << endl;
  int mid = low + ((high) / 2);
  if (high > 1) {

    // If found at mid, then return it
    if (array[mid] == x)
      return mid;

    // Search the left half
    if (array[mid] > x)
      return binarySearch(array, x, low, mid-low);

    // Search the right half
    else return binarySearch(array, x, mid, high-(mid-low));
  }

  return low;
}

void MsMergeSequential(int *out, int *in, long begin1, long end1, long begin2, long end2, long outBegin);

/**
  * the function performs a parallelized merge step 
  * @param t, the array the passed chunks of which are to merge
  * @param a, the output array 
  * @param p1, the start index of the left chunk
  * @param r1, the end index of the left chunk
  * @param p2, the start index of the right chunk
  * @param r2, the end index of the right chunk
  * @param p3
  */
void MergePar(int* t, int *a, int p1, int r1, int p2, int r2, int p3 ){
	int tp;
	if ( r1 > r2 ) //swap for std op later
	{
		tp = p1;
		p1 = p2;
		p2 = tp;
		tp = r1;
		r1 = r2;
		r2 = tp;
	}
	if ( r2 == 0 ) return;
	
    // check if chunks are big enough still for parallel merging
	if(r2>2000){
    	int q1 = p1+((r1)/2);
    	int q2 = q1;
    
    	if ( r1 > 0 ){
                    q2 = binarySearch(t, t[q1], p2, r2 );
            }
    
    	bool correspondent = true;
    	if(t[q1]<t[q2] && r1 > 0){
    		correspondent = false;
    		a[p3]=t[p1];
    		MergePar(t, a, p1+1, r1-1, p2, r2, p3+1);
    	}
    	if(correspondent){
        	if(q1>p1){
        		while(q1>p1 && t[q1-1]>t[q2]){
        			q1--;
        		}
        	}
        	int q3 = p3 + ( q1 - p1 ) + ( q2 - p2 );
        	a[q3] = t[q2];
        	#pragma omp taskgroup
        	{  	
        		#pragma omp task
        		MergePar( t, a, p1, q1-p1, p2, q2-p2, p3 );	//treat upper bounds as a size and not an index (pass q1, not q1-1)
        		#pragma omp task
        		MergePar( t, a, q1, r1-(q1-p1), q2+1, r2-(q2-p2)-1, q3+1);
        	}
    	}
	}else{ // chunks too small, merge sequentially
		MsMergeSequential(a, t, p1, p1+r1, p2, p2+r2, p3); 
	}
}

/**
  * sequential merge step (straight-forward implementation)
  * @param in, the array the passed chunks of which are to merge
  * @param out, the output array 
  * @param begin1, the start index of the left chunk
  * @param end1, the end index of the left chunk
  * @param begin1, the start index of the right chunk
  * @param end1, the end index of the right chunk
  * @param outbegin
  */
void MsMergeSequential(int *out, int *in, long begin1, long end1, long begin2, long end2, long outBegin) {

	long left = begin1;
	long right = begin2;

	long idx = outBegin;
	double runtime = get_time();
	while (left < end1 && right < end2) {

		if (in[left] <= in[right]) {
			out[idx] = in[left];
			left++;
		} else {
			out[idx] = in[right];
			right++;
		}
		idx++;
	}

	#pragma omp parallel
	{
	//printf("Num of threads: %d", omp_get_num_threads());
	__assume_aligned(out, 64);
	__assume_aligned(in, 64);
	#pragma vector aligned
	#pragma ivdep
	#pragma omp for schedule(static, 64)
	for (int i=left;i < end1;i++) {
		out[idx] = in[i];
		//left++, 
		idx++;
	}
	__assume_aligned(out, 64);
    __assume_aligned(in, 64);
    #pragma vector aligned
    #pragma ivdep
	#pragma omp for schedule(static, 64)
	for (int j=right; j < end2; j++) {
		out[idx] = in[j];
		//right++, 
		idx++;
	}
	}
	//runtime = get_time() - runtime;
	//printf("Time Elapsed for lines 117-136: %f s\n", runtime);

}

/**
  * sequential MergeSort applying Quicksort algorithm
  */
void MsSequential_noTasks(int *array, int *tmp, bool inplace, long begin, long end) {

	if (begin < (end - 1)) 
		quickSort(array,begin,end-1);

}


/**
  * MergeSort realized with OpenMP tasks
  * @param arr, the array (chunk) to sort
  * @param tmp, the temporary storage required when sorting out-of-place
  * @param inplace, whether to sort in place
  * @param begin, the start index of the chunk
  * @param end, the end index of the chunk
  * @param cutoff, the maximum size of the chunk to perform std::sort for
  */
void MsSequential(int *array, int *tmp, bool inplace, long begin, long end, int cutoff) {
	
	// check if chunk is big enough for task creation
	if(end-begin < cutoff){ // chunk too small, sort sequentially
		if(inplace) std::sort(array + begin, array + end);
		else{
			std::copy(array + begin, array + end, tmp + begin);
    		std::sort(tmp + begin, tmp + end);
		}
	}
	else{ // create tasks and sort parallely	
		if (begin < (end - 1)) {

			const long half = (begin + ((end-begin)/2));
			
			//create tasks such that they are not tied to the creating node and share the array as a ressource
			#pragma omp taskgroup
			{
				#pragma omp task shared(array) untied //affinity(array[begin:(half-begin)]) 
				MsSequential(array, tmp, !inplace, begin, half, cutoff);

				#pragma omp task shared(array) untied //affinity(array[half:(end-half)])
				MsSequential(array, tmp, !inplace, half, end, cutoff);
				//#pragma omp taskyield
			}
			
			// merge the sorted halves in a parallel manner
			if (inplace) {
				MergePar(tmp, array, begin, half-begin, half, end-half, begin);

			} else{
				MergePar(array, tmp, begin, half-begin, half, end-half, begin);

			}
		} else if (!inplace)tmp[begin] = array[begin];
		

	}
	
}


/**
  * Function initiates MsSequential without parallelization taking effect yet
  
  */
void MsSerial(int *array, int *tmp, const size_t size, int cutoff) {
	
	#pragma omp single
	MsSequential(array, tmp, true, 0, size, cutoff);
	
}


/** 
  * @brief program entry point
  */
int main(int argc, char* argv[]) {
	// variables to measure the elapsed time
	struct timeval t1, t2;
	double etime;

	// expect one or two command line arguments: array size [cutoff]
	if (argc < 2) {
		printf("Usage: MergeSort.exe <array size> [cutoff]\n");
		printf("\n");
		return EXIT_FAILURE;
	}
	else {
		const size_t stSize = strtol(argv[1], NULL, 10);
		int *data = (int*) numa_alloc_onnode(stSize * sizeof(int), 6);
		int *tmp = (int*) numa_alloc_onnode(stSize * sizeof(int), 6);
		int *ref = (int*) malloc(stSize * sizeof(int));
		int t=0;
		
		#pragma omp parallel
		t = omp_get_num_threads();

		size_t cutoff = 0.001*stSize;
		if(argc > 2) {
			cutoff = strtol(argv[2], NULL, 10);
		}

		printf("Initialization... \n");

		srand(95);
        
        // parallel initialization to enhance data locality for later on
		#pragma omp parallel for schedule(static, 32) num_threads(192) proc_bind(spread)
		for (size_t idx = 0; idx < stSize; ++idx){

			data[idx] = (int) (stSize * (double(rand()) / RAND_MAX));
		}
		std::copy(data, data + stSize, ref);

		double dSize = (stSize * sizeof(int)) / 1024 / 1024;
		printf("Sorting %zu elements of type int (%f MiB)...\n", stSize, dSize);

		gettimeofday(&t1, NULL);
		
        // start computing with 192 threads
		#pragma omp parallel num_threads(192) proc_bind(spread)
		MsSerial(data, tmp, stSize, cutoff);
		
		gettimeofday(&t2, NULL);
		etime = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000;
		etime = etime / 1000;

		printf("done, took %f sec. Verification...", etime);
		if (isSorted(ref, data, stSize)) {
			printf(" successful.\n");
		}
		else {
			printf(" FAILED.\n");
		}
		//cout << endl;
		//printArray(ref, stSize);
		/*int compArray[stSize];
		for(size_t i = 0; i < stSize; i++){
			if(ref[i]==data[i]) compArray[i]=1;
			else compArray[i]=0;
		}*/
		//printArray(compArray, stSize);
		
		// free memory on the heap
		numa_free(data, stSize);
		numa_free(tmp, stSize);
		free(ref);
	}

	return EXIT_SUCCESS;
}
