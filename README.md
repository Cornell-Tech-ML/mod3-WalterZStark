# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# Submission:

## Parallel Analyitics 

```
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/home/walter/Desktop/workspace/mod3-WalterZStark/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /home/walter/Desktop/workspace/mod3-WalterZStark/minitorch/fast_ops.py (163) 
-----------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                  | 
        out: Storage,                                                                          | 
        out_shape: Shape,                                                                      | 
        out_strides: Strides,                                                                  | 
        in_storage: Storage,                                                                   | 
        in_shape: Shape,                                                                       | 
        in_strides: Strides,                                                                   | 
    ) -> None:                                                                                 | 
        if np.array_equal(out_strides, in_strides) and np.array_equal(out_shape, in_shape):    | 
            for i in prange(len(out)):---------------------------------------------------------| #0
                out[i] = fn(in_storage[i])                                                     | 
            return                                                                             | 
                                                                                               | 
                                                                                               | 
        for ordinal in prange(len(out)):-------------------------------------------------------| #1
            input_index: Index = np.empty(len(in_shape), dtype=np.int32)                       | 
            output_index: Index = np.empty(len(out_shape), dtype=np.int32)                     | 
            to_index(ordinal, out_shape, output_index)                                         | 
            broadcast_index(output_index, out_shape, in_shape, input_index)                    | 
            output_pos = index_to_position(output_index, out_strides)                          | 
            input_pos = index_to_position(input_index, in_strides)                             | 
            out[output_pos] = fn(in_storage[input_pos])                                        | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/home/walter/Desktop/workspace/mod3-WalterZStark/minitorch/fast_ops.py (178) is 
hoisted out of the parallel loop labelled #1 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: input_index: Index = np.empty(len(in_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/home/walter/Desktop/workspace/mod3-WalterZStark/minitorch/fast_ops.py (179) is 
hoisted out of the parallel loop labelled #1 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: output_index: Index = np.empty(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/home/walter/Desktop/workspace/mod3-WalterZStark/minitorch/fast_ops.py (212)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /home/walter/Desktop/workspace/mod3-WalterZStark/minitorch/fast_ops.py (212) 
------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                   | 
        out: Storage,                                                                           | 
        out_shape: Shape,                                                                       | 
        out_strides: Strides,                                                                   | 
        a_storage: Storage,                                                                     | 
        a_shape: Shape,                                                                         | 
        a_strides: Strides,                                                                     | 
        b_storage: Storage,                                                                     | 
        b_shape: Shape,                                                                         | 
        b_strides: Strides,                                                                     | 
    ) -> None:                                                                                  | 
                                                                                                | 
                                                                                                | 
        if (                                                                                    | 
            np.array_equal(out_strides, a_strides)                                              | 
            and np.array_equal(out_strides, b_strides)                                          | 
            and np.array_equal(out_shape, a_shape)                                              | 
            and np.array_equal(out_shape, b_shape)                                              | 
        ):                                                                                      | 
            for i in prange(len(out)):----------------------------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                                         | 
            return                                                                              | 
                                                                                                | 
                                                                                                | 
        # Loop through each ordinal of the output                                               | 
        for ordinal in prange(len(out)):--------------------------------------------------------| #3
            a_input_index: Index = np.empty(len(a_shape), dtype=np.int32)                       | 
            b_input_index: Index = np.empty(len(b_shape), dtype=np.int32)                       | 
            output_index: Index = np.empty(len(out_shape), dtype=np.int32)                      | 
            # Given the ordinal, find its corresponding output index                            | 
            to_index(ordinal, out_shape, output_index)                                          | 
            # Find the corresponding input indices using broadcasting                           | 
            broadcast_index(output_index, out_shape, a_shape, a_input_index)                    | 
            broadcast_index(output_index, out_shape, b_shape, b_input_index)                    | 
            # Find corresponding positions to the indices                                       | 
            output_pos = index_to_position(output_index, out_strides)                           | 
            a_pos = index_to_position(a_input_index, a_strides)                                 | 
            b_pos = index_to_position(b_input_index, b_strides)                                 | 
            # Calculate the value of the output with a function applied to the correct input    | 
            out[output_pos] = fn(a_storage[a_pos], b_storage[b_pos])                            | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/home/walter/Desktop/workspace/mod3-WalterZStark/minitorch/fast_ops.py (238) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: a_input_index: Index = np.empty(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/home/walter/Desktop/workspace/mod3-WalterZStark/minitorch/fast_ops.py (239) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: b_input_index: Index = np.empty(len(b_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/home/walter/Desktop/workspace/mod3-WalterZStark/minitorch/fast_ops.py (240) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: output_index: Index = np.empty(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/home/walter/Desktop/workspace/mod3-WalterZStark/minitorch/fast_ops.py (277)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /home/walter/Desktop/workspace/mod3-WalterZStark/minitorch/fast_ops.py (277) 
----------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                        | 
        out: Storage,                                                                   | 
        out_shape: Shape,                                                               | 
        out_strides: Strides,                                                           | 
        a_storage: Storage,                                                             | 
        a_shape: Shape,                                                                 | 
        a_strides: Strides,                                                             | 
        reduce_dim: int,                                                                | 
    ) -> None:                                                                          | 
                                                                                        | 
        for i in prange(len(out)):------------------------------------------------------| #4
            reduce_size = a_shape[reduce_dim]                                           | 
            out_index: Index = np.empty(len(out_shape), dtype=np.int32)                 | 
            to_index(i, out_shape, out_index)                                           | 
            o = index_to_position(out_index, out_strides)                               | 
            out_index[reduce_dim] = 0                                                   | 
            current = fn(out[o], a_storage[index_to_position(out_index, a_strides)])    | 
                                                                                        | 
            for s in range(1, reduce_size):                                             | 
                out_index[reduce_dim] = s                                               | 
                storage_pose = 0                                                        | 
                for dim in range(len(a_shape)):                                         | 
                    storage_pose += out_index[dim] * a_strides[dim]                     | 
                current = fn(current, a_storage[storage_pose])                          | 
                                                                                        | 
            out[o] = current                                                            | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/home/walter/Desktop/workspace/mod3-WalterZStark/minitorch/fast_ops.py (289) is 
hoisted out of the parallel loop labelled #4 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.empty(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/home/walter/Desktop/workspace/mod3-WalterZStark/minitorch/fast_ops.py (308)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /home/walter/Desktop/workspace/mod3-WalterZStark/minitorch/fast_ops.py (308) 
----------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                  | 
    out: Storage,                                                                             | 
    out_shape: Shape,                                                                         | 
    out_strides: Strides,                                                                     | 
    a_storage: Storage,                                                                       | 
    a_shape: Shape,                                                                           | 
    a_strides: Strides,                                                                       | 
    b_storage: Storage,                                                                       | 
    b_shape: Shape,                                                                           | 
    b_strides: Strides,                                                                       | 
) -> None:                                                                                    | 
    """NUMBA tensor matrix multiply function.                                                 | 
                                                                                              | 
    Should work for any tensor shapes that broadcast as long as                               | 
                                                                                              | 
    ```                                                                                       | 
    assert a_shape[-1] == b_shape[-2]                                                         | 
    ```                                                                                       | 
                                                                                              | 
    Optimizations:                                                                            | 
                                                                                              | 
    * Outer loop in parallel                                                                  | 
    * No index buffers or function calls                                                      | 
    * Inner loop should have no global writes, 1 multiply.                                    | 
                                                                                              | 
                                                                                              | 
    Args:                                                                                     | 
    ----                                                                                      | 
        out (Storage): storage for `out` tensor                                               | 
        out_shape (Shape): shape for `out` tensor                                             | 
        out_strides (Strides): strides for `out` tensor                                       | 
        a_storage (Storage): storage for `a` tensor                                           | 
        a_shape (Shape): shape for `a` tensor                                                 | 
        a_strides (Strides): strides for `a` tensor                                           | 
        b_storage (Storage): storage for `b` tensor                                           | 
        b_shape (Shape): shape for `b` tensor                                                 | 
        b_strides (Strides): strides for `b` tensor                                           | 
                                                                                              | 
    Returns:                                                                                  | 
    -------                                                                                   | 
        None : Fills in `out`                                                                 | 
                                                                                              | 
    """                                                                                       | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                    | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                    | 
                                                                                              | 
                                                                                              | 
    # Loop over the batches in the output                                                     | 
    for batch in prange(out_shape[0]):--------------------------------------------------------| #5
        a_offset = batch * a_batch_stride                                                     | 
        b_offset = batch * b_batch_stride                                                     | 
        # Loop over every row and column of the output                                        | 
        for i in range(out_shape[1]):                                                         | 
            for j in range(out_shape[2]):                                                     | 
                # Initialize output value and index                                           | 
                dot = 0.0                                                                     | 
                out_idx = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]    | 
                # Find the dot product                                                        | 
                for k in range(a_shape[2]):                                                   | 
                    a_idx = a_offset + i * a_strides[1] + k * a_strides[2]                    | 
                    b_idx = b_offset + k * b_strides[1] + j * b_strides[2]                    | 
                    dot += a_storage[a_idx] * b_storage[b_idx]                                | 
                out[out_idx] = dot                                                            | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

```

## GPU

### Simple
`python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.01`
```

     Epoch      |         Loss         |     Correct     |    Time/Epoch (s)   
       0        |       5.767569       |       36        |        1.969        
      10        |       5.055773       |       45        |        0.900        
      20        |       4.317765       |       45        |        0.895        
      30        |       3.142304       |       47        |        0.878        
      40        |       2.573963       |       47        |        0.858        
      50        |       3.639232       |       49        |        0.843        
      60        |       2.592215       |       50        |        0.828        
      70        |       3.890807       |       50        |        0.812        
      80        |       4.097129       |       50        |        0.799        
      90        |       1.928244       |       50        |        0.791        
      100       |       2.028171       |       50        |        0.783        
      110       |       1.926196       |       50        |        0.778        
      120       |       2.485073       |       50        |        0.777        
      130       |       1.376387       |       49        |        0.787        
      140       |       1.716277       |       50        |        0.820        
      150       |       2.219794       |       50        |        0.839        
      160       |       1.024864       |       50        |        0.858        
      170       |       0.874950       |       50        |        0.875        
      180       |       1.424775       |       50        |        0.899        
      190       |       1.860732       |       50        |        0.914        
      200       |       1.540295       |       50        |        0.933        
      210       |       1.225240       |       50        |        0.947        
      220       |       1.012089       |       49        |        0.960        
      230       |       1.506478       |       50        |        0.977        
      240       |       1.701806       |       50        |        0.994        
      250       |       1.140615       |       50        |        1.005        
      260       |       1.471187       |       49        |        1.009        
      270       |       0.801179       |       49        |        1.021        
      280       |       0.849353       |       50        |        1.034        
      290       |       1.214085       |       50        |        1.042        
      300       |       1.374763       |       49        |        1.047        
      310       |       0.924684       |       50        |        1.046        
      320       |       1.607064       |       50        |        1.046        
      330       |       1.719017       |       49        |        1.054        
      340       |       1.354272       |       49        |        1.053        
      350       |       0.818776       |       50        |        1.052        
      360       |       0.642525       |       50        |        1.045        
      370       |       1.100054       |       50        |        1.039        
      380       |       1.440112       |       50        |        1.032        
      390       |       0.818630       |       49        |        1.025        
      400       |       1.182700       |       49        |        1.023        
      410       |       0.977650       |       50        |        1.021        
      420       |       0.875736       |       49        |        1.018        
      430       |       1.510113       |       50        |        1.013        
      440       |       0.731583       |       50        |        1.009        
      450       |       0.871779       |       50        |        1.007        
      460       |       0.804371       |       49        |        1.002        
      470       |       1.201871       |       50        |        0.996        
      480       |       1.110255       |       49        |        0.991        
      490       |       1.047424       |       50        |        0.989
```

### Split
`python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.025`
```
     Epoch      |         Loss         |     Correct     |    Time/Epoch (s)   
       0        |       6.856094       |       31        |        2.029        
      10        |       5.801207       |       40        |        0.843        
      20        |       3.831146       |       42        |        0.817        
      30        |       5.773163       |       43        |        0.811        
      40        |       4.228561       |       43        |        0.797        
      50        |       4.144891       |       43        |        0.787        
      60        |       4.006380       |       43        |        0.778        
      70        |       4.312440       |       47        |        0.774        
      80        |       3.788678       |       46        |        0.775        
      90        |       3.354941       |       47        |        0.772        
      100       |       3.387168       |       46        |        0.769        
      110       |       3.233149       |       48        |        0.772        
      120       |       2.152952       |       47        |        0.777        
      130       |       4.108102       |       49        |        0.778        
      140       |       3.232674       |       49        |        0.778        
      150       |       1.509397       |       49        |        0.778        
      160       |       1.989268       |       50        |        0.775        
      170       |       2.962576       |       49        |        0.773        
      180       |       1.489093       |       48        |        0.773        
      190       |       1.970444       |       49        |        0.772        
      200       |       2.417883       |       48        |        0.770        
      210       |       2.390565       |       50        |        0.769        
      220       |       0.864813       |       49        |        0.769        
      230       |       2.656718       |       50        |        0.768        
      240       |       1.083278       |       48        |        0.775        
      250       |       1.852729       |       50        |        0.775        
      260       |       1.094597       |       50        |        0.779        
      270       |       1.197163       |       50        |        0.784        
      280       |       1.367551       |       50        |        0.786        
      290       |       1.019745       |       48        |        0.788        
      300       |       0.808356       |       48        |        0.787        
      310       |       1.862627       |       50        |        0.790        
      320       |       1.582214       |       50        |        0.791        
      330       |       0.897965       |       50        |        0.793        
      340       |       0.980439       |       48        |        0.798        
      350       |       1.138817       |       50        |        0.802        
      360       |       1.007054       |       48        |        0.803        
      370       |       0.785398       |       50        |        0.805        
      380       |       2.189906       |       50        |        0.806        
      390       |       1.480352       |       49        |        0.805        
      400       |       0.647021       |       50        |        0.806        
      410       |       0.612712       |       49        |        0.807        
      420       |       0.617458       |       50        |        0.805        
      430       |       0.647425       |       50        |        0.807        
      440       |       0.606778       |       50        |        0.807        
      450       |       0.971656       |       49        |        0.809        
      460       |       0.695525       |       49        |        0.811        
      470       |       0.639620       |       47        |        0.814        
      480       |       1.425026       |       48        |        0.814        
      490       |       0.851739       |       49        |        0.818
```
### XOR

`python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.01`
```
     Epoch      |         Loss         |     Correct     |    Time/Epoch (s)   
       0        |       7.273509       |       34        |        1.870        
      10        |       6.717589       |       38        |        0.893        
      20        |       4.730246       |       39        |        0.843        
      30        |       6.078783       |       36        |        0.813        
      40        |       5.831928       |       38        |        0.811        
      50        |       4.226905       |       43        |        0.811        
      60        |       4.701477       |       43        |        0.810        
      70        |       4.476205       |       45        |        0.805        
      80        |       4.115549       |       45        |        0.803        
      90        |       4.354209       |       45        |        0.796        
      100       |       3.939519       |       45        |        0.788        
      110       |       5.001491       |       47        |        0.783        
      120       |       3.837497       |       45        |        0.780        
      130       |       3.702483       |       46        |        0.787        
      140       |       3.532434       |       47        |        0.803        
      150       |       2.865386       |       47        |        0.814        
      160       |       3.769073       |       47        |        0.821        
      170       |       3.422963       |       47        |        0.818        
      180       |       3.920518       |       47        |        0.817        
      190       |       2.964287       |       47        |        0.816        
      200       |       3.342990       |       48        |        0.816        
      210       |       3.455516       |       47        |        0.812        
      220       |       2.263517       |       48        |        0.813        
      230       |       2.648476       |       47        |        0.822        
      240       |       2.706724       |       47        |        0.834        
      250       |       4.022263       |       47        |        0.839        
      260       |       2.723536       |       48        |        0.843        
      270       |       2.747733       |       47        |        0.848        
      280       |       1.605990       |       48        |        0.854        
      290       |       2.547571       |       49        |        0.858        
      300       |       3.074734       |       47        |        0.863        
      310       |       3.548178       |       49        |        0.868        
      320       |       1.809111       |       48        |        0.871        
      330       |       1.915495       |       49        |        0.875        
      340       |       2.098011       |       49        |        0.879        
      350       |       2.808306       |       48        |        0.884        
      360       |       1.798811       |       48        |        0.886        
      370       |       2.098139       |       48        |        0.888        
      380       |       1.148329       |       48        |        0.887        
      390       |       2.128571       |       49        |        0.884        
      400       |       1.748015       |       48        |        0.880        
      410       |       3.140790       |       49        |        0.877        
      420       |       1.795037       |       48        |        0.873        
      430       |       1.764221       |       49        |        0.869        
      440       |       2.156108       |       49        |        0.866        
      450       |       1.746143       |       48        |        0.863        
      460       |       2.774231       |       49        |        0.859        
      470       |       1.960235       |       48        |        0.856        
      480       |       1.008110       |       49        |        0.853        
      490       |       1.044537       |       48        |        0.851
```

### Bigger Split (500 Hidden Layers)



## CPU

### Simple

`python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.03`
```
     Epoch      |         Loss         |     Correct     |    Time/Epoch (s)   
       0        |       6.950657       |       29        |        9.060        
      10        |       3.314151       |       47        |        0.889        
      20        |       2.429532       |       48        |        0.503        
      30        |       2.039655       |       48        |        0.368        
      40        |       3.308067       |       48        |        0.301        
      50        |       2.324356       |       49        |        0.260        
      60        |       1.801424       |       49        |        0.233        
      70        |       1.980883       |       49        |        0.213        
      80        |       0.784468       |       49        |        0.198        
      90        |       1.342311       |       50        |        0.187        
      100       |       0.648903       |       50        |        0.177        
      110       |       1.165824       |       50        |        0.170        
      120       |       1.568274       |       49        |        0.163        
      130       |       0.765445       |       50        |        0.158        
      140       |       1.207526       |       50        |        0.153        
      150       |       1.066485       |       50        |        0.149        
      160       |       0.183455       |       50        |        0.146        
      170       |       0.532236       |       50        |        0.143        
      180       |       1.118006       |       48        |        0.140        
      190       |       0.894151       |       50        |        0.137        
      200       |       0.949150       |       50        |        0.135        
      210       |       1.161860       |       50        |        0.133        
      220       |       1.415663       |       49        |        0.140        
      230       |       0.555416       |       50        |        0.141        
      240       |       0.480061       |       50        |        0.139        
      250       |       0.877371       |       49        |        0.137        
      260       |       0.446911       |       50        |        0.135        
      270       |       1.861600       |       49        |        0.133        
      280       |       0.958024       |       50        |        0.131        
      290       |       0.101288       |       50        |        0.131        
      300       |       1.069498       |       50        |        0.130        
      310       |       0.026271       |       50        |        0.129        
      320       |       1.398954       |       50        |        0.127        
      330       |       0.621732       |       50        |        0.126        
      340       |       0.437379       |       50        |        0.126        
      350       |       0.577208       |       50        |        0.125        
      360       |       0.503954       |       50        |        0.124        
      370       |       0.400439       |       50        |        0.124        
      380       |       0.416044       |       50        |        0.124        
      390       |       0.305709       |       50        |        0.123        
      400       |       0.148474       |       50        |        0.122        
      410       |       0.935701       |       50        |        0.121        
      420       |       0.092669       |       50        |        0.121        
      430       |       0.057286       |       50        |        0.121        
      440       |       1.922671       |       48        |        0.121        
      450       |       0.522280       |       50        |        0.121        
      460       |       0.876962       |       50        |        0.121        
      470       |       0.752177       |       50        |        0.120        
      480       |       0.015879       |       50        |        0.119        
      490       |       0.130118       |       50        |        0.119
```

### Split
`python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.03`
```
     Epoch      |         Loss         |     Correct     |    Time/Epoch (s)   
       0        |       5.251718       |       31        |        9.001        
      10        |       6.675708       |       40        |        0.862        
      20        |       6.070984       |       38        |        0.475        
      30        |       4.790402       |       44        |        0.337        
      40        |       4.613306       |       42        |        0.266        
      50        |       2.905904       |       40        |        0.224        
      60        |       4.377694       |       46        |        0.195        
      70        |       2.921401       |       43        |        0.174        
      80        |       3.112964       |       43        |        0.158        
      90        |       2.603167       |       46        |        0.146        
      100       |       1.225976       |       48        |        0.136        
      110       |       2.123058       |       45        |        0.129        
      120       |       2.061381       |       43        |        0.122        
      130       |       2.818310       |       48        |        0.116        
      140       |       1.476648       |       50        |        0.111        
      150       |       1.734310       |       49        |        0.107        
      160       |       1.999682       |       45        |        0.103        
      170       |       1.189989       |       50        |        0.100        
      180       |       1.757301       |       50        |        0.097        
      190       |       1.836990       |       50        |        0.095        
      200       |       1.737276       |       50        |        0.092        
      210       |       1.580983       |       47        |        0.090        
      220       |       1.266096       |       50        |        0.088        
      230       |       1.276753       |       50        |        0.087        
      240       |       1.255047       |       49        |        0.085        
      250       |       2.233965       |       48        |        0.083        
      260       |       1.628525       |       50        |        0.082        
      270       |       1.315579       |       50        |        0.081        
      280       |       1.474166       |       49        |        0.080        
      290       |       0.787188       |       50        |        0.079        
      300       |       1.577906       |       49        |        0.078        
      310       |       0.746597       |       48        |        0.077        
      320       |       0.686019       |       50        |        0.076        
      330       |       0.563638       |       50        |        0.075        
      340       |       0.548554       |       50        |        0.075        
      350       |       0.597108       |       50        |        0.074        
      360       |       0.378764       |       48        |        0.074        
      370       |       0.708761       |       50        |        0.074        
      380       |       0.484243       |       50        |        0.073        
      390       |       0.699976       |       50        |        0.073        
      400       |       0.968223       |       49        |        0.073        
      410       |       0.901577       |       48        |        0.072        
      420       |       0.775858       |       50        |        0.072        
      430       |       0.513345       |       50        |        0.072        
      440       |       0.447226       |       50        |        0.071        
      450       |       0.916232       |       50        |        0.071        
      460       |       0.400114       |       50        |        0.071        
      470       |       0.230092       |       50        |        0.071        
      480       |       0.330108       |       50        |        0.071        
      490       |       0.573910       |       50        |        0.070
```

### XOR

`python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.03`
```
     Epoch      |         Loss         |     Correct     |    Time/Epoch (s)   
       0        |       7.952607       |       34        |        10.927       
      10        |       6.452900       |       35        |        1.069        
      20        |       5.809179       |       43        |        0.605        
      30        |       5.524318       |       43        |        0.439        
      40        |       3.387032       |       44        |        0.354        
      50        |       3.259050       |       44        |        0.303        
      60        |       3.419395       |       44        |        0.268        
      70        |       3.037042       |       47        |        0.243        
      80        |       2.400119       |       47        |        0.225        
      90        |       3.053398       |       47        |        0.210        
      100       |       2.212108       |       46        |        0.198        
      110       |       2.619909       |       47        |        0.189        
      120       |       0.817636       |       47        |        0.180        
      130       |       2.240407       |       47        |        0.173        
      140       |       2.033473       |       49        |        0.168        
      150       |       2.309163       |       47        |        0.164        
      160       |       1.004503       |       47        |        0.161        
      170       |       1.415463       |       47        |        0.156        
      180       |       1.914798       |       47        |        0.152        
      190       |       1.239233       |       47        |        0.148        
      200       |       0.790842       |       47        |        0.145        
      210       |       2.584014       |       47        |        0.142        
      220       |       2.300279       |       49        |        0.139        
      230       |       1.095459       |       47        |        0.136        
      240       |       1.134537       |       47        |        0.133        
      250       |       2.604703       |       49        |        0.129        
      260       |       2.041411       |       48        |        0.127        
      270       |       0.647177       |       49        |        0.124        
      280       |       1.277644       |       49        |        0.125        
      290       |       2.024897       |       49        |        0.124        
      300       |       2.145531       |       49        |        0.123        
      310       |       1.199021       |       48        |        0.124        
      320       |       1.357022       |       49        |        0.122        
      330       |       0.491475       |       49        |        0.121        
      340       |       1.821601       |       50        |        0.120        
      350       |       0.265302       |       47        |        0.119        
      360       |       0.445860       |       49        |        0.119        
      370       |       0.489538       |       49        |        0.119        
      380       |       1.638549       |       49        |        0.119        
      390       |       0.907832       |       49        |        0.119        
      400       |       1.638995       |       49        |        0.120        
      410       |       0.906588       |       49        |        0.122        
      420       |       0.132157       |       49        |        0.123        
      430       |       1.471211       |       49        |        0.123        
      440       |       0.867883       |       48        |        0.123        
      450       |       1.059170       |       48        |        0.123        
      460       |       0.745835       |       49        |        0.123        
      470       |       2.091997       |       49        |        0.123        
      480       |       0.730195       |       49        |        0.123        
      490       |       1.499802       |       50        |        0.123 
```

### Bigger Split (500 Hidden Layers)

`python run_fast_tensor.py --BACKEND cpu --HIDDEN 500 --DATASET split --RATE 0.03`
```
     Epoch      |         Loss         |     Correct     |    Time/Epoch (s)   
       0        |      27.630553       |       29        |        9.473        
      10        |      82.777194       |       29        |        1.270        
      20        |      38.264976       |       21        |        0.911        
      30        |       0.653728       |       47        |        0.782        
      40        |       0.592061       |       48        |        0.711        
      50        |       1.870721       |       49        |        0.651        
      60        |       0.194176       |       49        |        0.619        
      70        |       0.543554       |       49        |        0.587        
      80        |       1.606673       |       44        |        0.571        
      90        |       0.859118       |       49        |        0.559        
      100       |       0.248500       |       49        |        0.543        
      110       |       0.695948       |       50        |        0.541        
      120       |       0.277762       |       50        |        0.540        
      130       |       1.746899       |       47        |        0.553        
      140       |       1.276828       |       48        |        0.557        
      150       |       0.905882       |       50        |        0.567        
      160       |       0.626468       |       50        |        0.575        
      170       |       0.156046       |       50        |        0.578        
      180       |       0.843994       |       50        |        0.581        
      190       |       0.800724       |       50        |        0.584        
      200       |       0.431586       |       50        |        0.583        
      210       |       0.094760       |       50        |        0.576        
      220       |       0.308124       |       50        |        0.567        
      230       |       0.153658       |       50        |        0.565        
      240       |       0.721038       |       50        |        0.565        
      250       |       0.607208       |       50        |        0.564        
      260       |       0.194973       |       50        |        0.563        
      270       |       0.433132       |       50        |        0.561        
      280       |       0.205897       |       50        |        0.565        
      290       |       0.260495       |       50        |        0.567        
      300       |       0.102717       |       50        |        0.569        
      310       |       0.051568       |       50        |        0.568        
      320       |       0.284204       |       50        |        0.571        
      330       |       0.087886       |       50        |        0.570        
      340       |       0.194479       |       50        |        0.573        
      350       |       0.024498       |       50        |        0.579        
      360       |       0.388972       |       50        |        0.582        
      370       |       0.144497       |       50        |        0.578        
      380       |       0.086429       |       50        |        0.575        
      390       |       0.331390       |       50        |        0.572        
      400       |       0.390405       |       50        |        0.569        
      410       |       0.041469       |       50        |        0.564        
      420       |       0.080123       |       50        |        0.562        
      430       |       0.431206       |       50        |        0.562        
      440       |       0.323664       |       50        |        0.562        
      450       |       0.070850       |       50        |        0.561        
      460       |       0.019740       |       50        |        0.559        
      470       |       0.067061       |       50        |        0.560        
      480       |       0.121904       |       50        |        0.558        
      490       |       0.069887       |       50        |        0.554
```