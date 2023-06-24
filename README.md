# tfnufft
TensorFlow2 Implementation of Non-Uniform FFT with Kaiser-Bessel Gridding

This package is an adaptation of nuFFT and adjoint nuFFT functions in [SigPy](https://github.com/mikgroup/sigpy).  
Currently supports two computation modes 
- **Exact kernel computation**:  
Kaiser-bessel kernel values are computed on the fly, no table interpolation is used.
- **Precomputed kernel**:  
Kaiser-bessel kernel values are first computed and stored. Table interpolation is not used for kernel computation. Precomputation must be repeated for each set of coordinates.

# Computation Time
The computation speeds are given in milliseconds, for a 256x256 image with a spokelength of 512 and 405 spokes. The numbers are obtained on an NVIDIA GeForce RTX 2080 Ti graphics processor.

### *Exact kernel computation*  

| Operation | CPU<br>(eager) | CPU<br>(graph mode) | GPU<br>(eager) | GPU<br>(graph mode) |
|:---------:|:--------------:|:-------------------:|:--------------:|:-------------------:|
|  Forward  |      395.1     |        259.3        |      111.7     |         13.3        |
|  Adjoint  |      568.8     |        438.1        |      119.3     |         14.1        |

### *Precomputed kernel*

|  Operation 	| CPU<br>(eager) 	| CPU<br>(graph mode) 	| GPU<br>(eager) 	| GPU<br>(graph mode) 	|
|:----------:	|:--------------:	|:-------------------:	|:--------------:	|:-------------------:	|
| Precompute 	|      15.8      	|         15.1        	|      12.4      	|         12.9        	|
|   Forward  	|      275.2     	|        249.5        	|      21.3      	|         12.8        	|
|   Adjoint  	|      429.1     	|        395.7        	|      23.9      	|         13.4        	|

The computation times can be obtained by running the following commands
<pre><code>python profile_tfnufft.py
python profile_tfnufft_precomputed.py
</code></pre>

# About
If you use this library for your work, please consider citing the following work:

    @misc{alkan2023autosamp,
          title={AutoSamp: Autoencoding MRI Sampling via Variational Information Maximization}, 
          author={Cagan Alkan and Morteza Mardani and Shreyas S. Vasanawala and John M. Pauly},
          year={2023},
          eprint={2306.02888},
          archivePrefix={arXiv},
          primaryClass={eess.IV}
    }
