#include "../ccml.h"

int main() {
    // creating new memory context
    int capacity = 2<<16;
    ccml_context * ctx = ccml_new_context(capacity);

    // creating 2d 2x2 tensor with fp32 type with gradient tracking
    ccml_tensor * x = ccml_new_tensor_2d(ctx, CCML_TYPE_FP32, 2, 2, true);
    ccml_tensor * z = ccml_mul(ctx, ccml_exp(ctx, ccml_sin(ctx, x)), ccml_sin(ctx, x));
    ccml_fill(ctx, x, 2.0f);

    // creating a new computational graph
    ccml_graph * graph = ccml_new_graph(ctx, z);
    ccml_graph_execute(graph);
    
    // freeing the context
    ccml_context_free(ctx);
}
