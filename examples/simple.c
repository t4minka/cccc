#include "../ccml.h"

int main() {
    // define 2d tensor x with fp32 data type, with gradient tracking
    ccml_tensor * x = ccml_new_tensor_2d(CCML_TYPE_FP32, 2, 3, true);
    // perform arithmetic operations
    ccml_tensor * y = ccml_sin(ccml_log(x));
    // create computational graph
    ccml_graph * graph = ccml_new_graph(y);
    // execute the graph
    ccml_graph_execute(graph, CCML_BACKEND_METAL);s
    // free the memory :)
    ccml_graph_free(graph);
}    