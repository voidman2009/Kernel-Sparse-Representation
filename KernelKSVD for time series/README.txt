function DTW and dp are designed for computing DTW distance
function ERP is for ERP distance, correspondingly, ERPTest introduces 1NN classifier to test the function
function TWED is for TWED distance, correspondingly, TWEDTest introduces 1NN classifier to test the function

The above three are the state-of-art similarity measurements of time series
------------------------------------------------------------------------------

function KOMP_ONE realizes sparse coding for a given dictionary (Kernel SRC model)
KOMP_ONETEST calls the above function to realize the Kernel SRC model

function KKSVD realizes the kernel dictionary learning model (Kernel KSVD)
KERNEL_KSVD invokes KOMP_ONE, KKSVD, which completes the iterative learning phase
------------------------------------------------------------------------------

SupervisedDL realizes Kernel LC_KSVD