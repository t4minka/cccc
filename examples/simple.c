#include "../ccml.h"

int main() {
    // define 2d tensor x with fp32 data type, with gradient tracking
    ccml_tensor * x = ccml_new_tensor_2d(CCML_TYPE_FP32, 2, 3, false);
    ccml_fill(x, 2.0f);
    ccml_tensor * y = ccml_new_tensor_2d(CCML_TYPE_FP32, 3, 4, false);
    ccml_fill(y, 3.0f);
    // perform arithmetical operations
    ccml_tensor * z = ccml_matmul(x, y);
    // create computational graph
    ccml_graph * graph = ccml_new_graph(z);
    // generate the kernel and metal setup code and run it
    ccml_graph_execute(graph);
    // free the memory :)
    ccml_graph_free(graph);
}