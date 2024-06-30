#define CCML_BACKEND_METAL
#include "../ccml.h"

int main() {
    // creating new memory context
    ccml_context * ctx = ccml_new_context(2<<16 /* bytes */);

    // creating 3d 2x3x4 tensor w/o gradient tracking
    ccml_tensor * x = ccml_new_tensor(ctx, 2, 3);
    ccml_tensor * z = ccml_sin(ctx, ccml_cos(ctx, x));

    // initialising tensors with data
    ccml_fill(ctx, x, 2.0f);

    // creating a new computational graph
    ccml_graph * graph = ccml_new_graph(ctx, z);
    ccml_graph_execute(ctx, graph);
    
    // freeing the context
    ccml_context_free(ctx);
}