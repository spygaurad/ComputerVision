
# Image Processing Conceptual Questions

## Point-Based Transformations

1. **Histogram Equalization Purpose**: Explain the primary goal of histogram equalization and describe the type of image that would benefit most from this transformation. How does it differ from contrast stretching?
2. **Cumulative Distribution Function**: In histogram equalization, why is the cumulative distribution function (CDF) used instead of just the histogram itself? What property of the CDF makes it suitable for intensity transformation?
3. **Gamma Transformation Application**: Compare the visual effects of applying a gamma transformation with γ < 1 versus γ > 1 on an underexposed photograph. Which transformation would be more appropriate and why?
4. **Intensity Range Mapping**: Given a grayscale image with intensity values concentrated in the range , design a linear scaling transformation to utilize the full dynamic range. Explain each parameter in your transformation formula.
5. **Histogram Analysis**: An image histogram shows two distinct peaks with a valley in between. What does this suggest about the image content? Would histogram equalization be beneficial for such an image? Justify your answer.

## Convolution and Linear Filtering

6. **Boundary Conditions**: Explain the three main boundary condition strategies (zero, replicated, periodic) used when applying convolution near image borders. Describe a scenario where each strategy would produce significantly different results.
7. **Filter Size Selection**: Why are convolution filters typically designed with odd dimensions (3x3, 5x5, etc.) rather than even dimensions? What practical problem does this solve?
8. **Convolution Commutativity**: Is image convolution a commutative operation? If I(x,y) * h(x,y) = h(x,y) * I(x,y), explain what this property means in practical image processing terms.
9. **Separable Filters**: A 2D Gaussian filter can be decomposed into two 1D filters applied sequentially. What computational advantage does this separability provide for a 7x7 filter compared to direct 2D convolution?
10. **Filter Normalization**: Why must the weights in a box filter or Gaussian filter sum to 1? What would happen to image brightness if the sum was greater than or less than 1?

## Smoothing and Denoising

11. **Box vs Gaussian Filter**: Compare the frequency response characteristics of a box filter and a Gaussian filter. Which one provides better smoothing properties and why?
12. **Gaussian Standard Deviation**: How does increasing the standard deviation (σ) of a Gaussian filter affect the resulting filtered image? What is the relationship between σ and the required filter size?
13. **Denoising Trade-offs**: Explain the fundamental trade-off between noise reduction and edge preservation when applying smoothing filters. Why can't we achieve both perfectly simultaneously?
14. **Multiple Filter Applications**: If you apply a 3x3 box filter twice to an image, is the result equivalent to applying a 5x5 box filter once? Explain the similarities and differences.
15. **Salt-and-Pepper Noise**: Why is a median filter more effective than a mean filter for removing salt-and-pepper noise? Explain using the properties of these statistical measures.

## Edge Detection

16. **Sobel Operator Design**: The Sobel operator combines smoothing and differentiation. Explain how the weights in the Sobel Sx kernel [-1,0,1; -2,0,2; -1,0,1] achieve both operations simultaneously.
17. **Gradient Magnitude**: Given horizontal (Gx) and vertical (Gy) gradient responses from Sobel filters, how would you compute the edge magnitude and direction? What information does each provide?
18. **First Derivative and Edges**: Why do first derivative operators (like Sobel) respond strongly to edges in an image? Relate this to the intensity profile across an edge boundary.
19. **Vertical vs Horizontal Edges**: Describe what type of edge features the Sobel Sx kernel would detect versus the Sobel Sy kernel. Provide an example of an image feature each would respond to strongly.
20. **Edge Detection Sensitivity**: Why must images typically be smoothed (e.g., with a Gaussian filter) before applying edge detection operators? What problem does pre-smoothing solve?

## Linear vs Nonlinear Filters

21. **Linearity Property**: Define what makes a filter "linear" using the mathematical property g(a·I₁ + b·I₂) = a·g(I₁) + b·g(I₂). Give examples of one linear and one nonlinear filter from the course material.
22. **Median Filter Properties**: Explain why the median filter is classified as a nonlinear filter. What advantage does this nonlinearity provide for certain types of noise?
23. **Edge Preservation**: Compare how mean filters and median filters affect edges differently. Which one better preserves sharp boundaries and why?
24. **Order Statistics**: The median filter is based on order statistics rather than weighted averaging. Explain how this fundamental difference leads to different noise suppression characteristics.
25. **Filter Selection Strategy**: Given an image with both Gaussian noise and impulse noise, would you choose a mean filter, median filter, or a combination? Justify your selection strategy.

## Advanced Concepts

26. **Sharpening Filter Design**: Explain how a sharpening filter [0,0,0; 0,2,0; 0,0,0] - [1,1,1; 1,1,1; 1,1,1]/9 works. Why does subtracting the blurred version enhance edges?
27. **Frequency Domain Interpretation**: Low-pass filters suppress high frequencies, while high-pass filters suppress low frequencies. Classify these operations: smoothing, edge detection, sharpening. Explain each classification.
28. **Identity Filter Effect**: What does applying the identity kernel (center element = 1, all others = 0) produce? In what debugging or processing pipeline scenario would this be useful?
29. **Transformation Window Size**: Compare the conceptual differences between point-wise (1×1), local (k×k), and global (m×n) intensity transformations. Give an example operation for each category.
30. **Computational Complexity**: For an M×N image and a k×k filter, what is the computational complexity of direct convolution? How many multiplication operations are required per output pixel?


