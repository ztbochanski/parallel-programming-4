# parallel-programming-SIMD

### Vectorized Array Multiplication and Multiplication/Reduction using SSE (Streaming SIMD Extensions)

### C implementation

Array multiplication
```c
for (int i = 0; i < n; i++)
{
	C[i] = A[i] * B[i];
}
```

Summation
```c
float sum = 0.0;

for (int i = 0; i < n; i++)
{
	sum += A[i] * B[i];
}

return sum;
```

### Commentary

1. Machine used: `Oregon State Flip: Linux`
2. Show the 2 tables of performances for each array size and the corresponding speedups
	- ![Screenshot 2023-05-11 at 19.17.51.png](https://github.com/ztbochanski/parallel-programming-SIMD/blob/f390b11fa12254b9cb4d4a1c609e55ddd6706799/Screenshot%202023-05-11%20at%2019.17.51.png)
3. Show the graphs (or graph) of SIMD/non-SIMD speedup versus array size (either one graph with two curves, or two graphs each with one curve)
	- ![Picture1 1.svg](https://github.com/ztbochanski/parallel-programming-SIMD/blob/008241cb862bcee0f42e43e7e4c650346affc0e8/Picture1%201.svg)
4. What patterns are you seeing in the speedups?
	- Based on the hardware and compiler used, there were a few observed patterns. 
		1. As array size increases there is an increase in speedup (however this is only initially)
		2. At a certain array size, speedup gains plateau.
		3. As array size continues to increase past the plateau there are diminishing returns in speedup.
		4. Past a certain point there is a significant decrease in speedup gains.
1. Are they consistent across a variety of array sizes?
	- The patterns for both summation and multiplication are consistent across similar array sizes, however, multiplication without reduction has a much larger performance drop.
1. Why or why not, do you think?
	1. For the initial speed increase, SSE does much better which makes sense because there can be multiple data elements getting simultaneously executed. This is how SSE works where one instruction can work on multiple pieces of data so this logically makes sense. The following patterns however are interesting because performance degradation occurs.
	2. For patterns from points 2 & 3 above, beyond the size of the data speedup gains diminish and the curve starts to plateau. This could indicate the overhead of transforming the scalar operations (one data point) to a vector operation or vectorization meaning the SIMD instructions that operate on multiple elements at once.
	3. Finally significant decreases in speedup gains could include the overhead discussed above as well as memory limitations. Data could potentially be waiting to be fetched from memory resulting in worse speedup.
	4. Below is a zoomed-in portion of the graph eliminating the 8M array size to better show the speedup increase and plateau before it starts to get worse:
	   - ![Picture2.svg](https://github.com/ztbochanski/parallel-programming-SIMD/blob/2cfdcb62800c019b1f5c181bfd07fd2e5f165660/Picture2.svg)
