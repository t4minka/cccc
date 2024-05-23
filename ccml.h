#if !defined(CCML_IMPL)
#define CCML_IMPL

#define CCML_SRCS_MAX 2
#define CCML_TYPE_MAX 3
#define CCML_VIEW_MAX 4
#define CCML_DIMS_MAX 4

#define CCML_KERN_MAX 16
#define CCML_CHAR_MAX 100
#define CCML_NODE_MAX 128

#define __STDC_WANT_IEC_60559_TYPES_EXT__
#include <float.h>
#include <math.h>
#include <stdbool.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199409L) && !defined(CCML_API)
    #define CCML_API static inline
#elif !defined(CCML_API)
    #define CCML_API static
#endif

#define CCML_ASSERT(x, ...) do { if (!(x)) {                           \
    fprintf(stderr, "CCML_ASSERT: %s:%d: %s", __FILE__, __LINE__, #x); \
    __VA_OPT__(fprintf(stderr, ", %s", __VA_ARGS__);)                  \
    exit(EXIT_FAILURE);                                                \
} } while(0)

// known issues:
// - including ccml.h in separate compilation units compiles separate/independent symbols
// - tensors of size 1 aren't supported, due to attempting to embed them into the kernel
// - a lot of function return statuses aren't checked, mostly snprintf/fread/fwrite
// - if a tensor isn't part of a graph it won't get freed by ccml_graph_free()

//
//  ████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗
//  ╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗
//     ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝
//     ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗
//     ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║
//     ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝
//

typedef enum ccml_backend {
    CCML_BACKEND_METAL
} ccml_backend;

typedef enum ccml_type {
    CCML_TYPE_FP32
} ccml_type;

const static int ccml_type_sizes[CCML_TYPE_MAX] = {
    [CCML_TYPE_FP32] = sizeof(float),
};

typedef enum ccml_buff {
    // no buffer, values only existing as intermediary scalars in the kernel
    CCML_BUFF_NONE,
    // dedicated buffer for constant values
    CCML_BUFF_CNST,
    // dedicated buffer for loaded values
    CCML_BUFF_LOAD,
    // dedicated buffer for saved values
    CCML_BUFF_SAVE
} ccml_buff;

typedef enum ccml_oper {
    CCML_OPER_NONE,
    CCML_OPER_LOG,
    CCML_OPER_EXP,
    CCML_OPER_SIN,
    CCML_OPER_REC,
    CCML_OPER_SQRT,
    CCML_OPER_ADD,
    CCML_OPER_MUL,
    CCML_OPER_SUM,
} ccml_oper;

typedef struct ccml_tensor {
    ccml_type type;
    ccml_buff buff;
    ccml_oper oper;

    int shape[CCML_VIEW_MAX][CCML_DIMS_MAX];
    int stride[CCML_VIEW_MAX][CCML_DIMS_MAX];

    // the view system for shape tracking is borrowed from tinygrad in stacking
    // multiple shape views, with tensor->view[0] being current tensor's last view
    // and tensor->view[1] and tensor->view[2] being the view of tensor's children
    int view[CCML_SRCS_MAX + 1];

    struct ccml_tensor * src[CCML_SRCS_MAX];
    struct ccml_tensor * grad;

    float * data;
    int index;
    bool requires_grad;
} ccml_tensor;

//
//  ██╗███╗   ██╗██╗████████╗
//  ██║████╗  ██║██║╚══██╔══╝
//  ██║██╔██╗ ██║██║   ██║
//  ██║██║╚██╗██║██║   ██║
//  ██║██║ ╚████║██║   ██║
//  ╚═╝╚═╝  ╚═══╝╚═╝   ╚═╝
//
//  ███████╗██╗   ██╗███╗   ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗███████╗
//  ██╔════╝██║   ██║████╗  ██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
//  █████╗  ██║   ██║██╔██╗ ██║██║        ██║   ██║██║   ██║██╔██╗ ██║███████╗
//  ██╔══╝  ██║   ██║██║╚██╗██║██║        ██║   ██║██║   ██║██║╚██╗██║╚════██║
//  ██║     ╚██████╔╝██║ ╚████║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║███████║
//  ╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
//

CCML_API ccml_tensor *
ccml_new_tensor_impl(ccml_type type, int * shape) {
    ccml_tensor * result = malloc(sizeof(ccml_tensor));
    int ne[CCML_DIMS_MAX] = {
        shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1
    };

    *result = (ccml_tensor){
        /*.type          =*/ type,
        /*.buff          =*/ CCML_BUFF_NONE,
        /*.oper          =*/ CCML_OPER_NONE,
        /*.shape         =*/ {[0] = {shape[0], shape[1], shape[2], shape[3]}},
        /*.stride        =*/ {[0] = {ne[0], ne[1], ne[2], ne[3]}},
        /*.view          =*/ {0},
        /*.src           =*/ {NULL},
        /*.grad          =*/ NULL,
        /*.data          =*/ NULL,
        /*.graph_index   =*/ -1,
        /*.requires_grad =*/ false,
    };

    return result;
}

CCML_API ccml_tensor *
ccml_new_tensor_1d(ccml_type type, int ne0, bool requires_grad) {
    int shape[CCML_DIMS_MAX] = {ne0, 1, 1, 1};

    ccml_tensor * result = ccml_new_tensor_impl(type, shape);
    result->buff = CCML_BUFF_LOAD;
    result->requires_grad = requires_grad;

    return result;
}

CCML_API ccml_tensor *
ccml_new_tensor_2d(ccml_type type, int ne0, int ne1, bool requires_grad) {
    int shape[CCML_DIMS_MAX] = {ne0, ne1, 1, 1};

    ccml_tensor * result = ccml_new_tensor_impl(type, shape);
    result->buff = CCML_BUFF_LOAD;
    result->requires_grad = requires_grad;

    return result;
}

CCML_API ccml_tensor *
ccml_new_tensor_3d(ccml_type type, int ne0, int ne1, int ne2, bool requires_grad) {
    int shape[CCML_DIMS_MAX] = {ne0, ne1, ne2, 1};

    ccml_tensor * result = ccml_new_tensor_impl(type, shape);
    result->buff = CCML_BUFF_LOAD;
    result->requires_grad = requires_grad;

    return result;
}

CCML_API ccml_tensor *
ccml_new_tensor_4d(ccml_type type, int ne0, int ne1, int ne2, int ne3, bool requires_grad) {
    int shape[CCML_DIMS_MAX] = {ne0, ne1, ne2, ne3};

    ccml_tensor * result = ccml_new_tensor_impl(type, shape);
    result->buff = CCML_BUFF_LOAD;
    result->requires_grad = requires_grad;

    return result;
}

CCML_API int ccml_size(ccml_tensor * tensor) {
    int v = tensor->view[0];
    return tensor->shape[v][0] * tensor->shape[v][1] *
           tensor->shape[v][2] * tensor->shape[v][3];
}

CCML_API void ccml_set(ccml_tensor * tensor, float * data) {
    tensor->type = CCML_TYPE_FP32;
    tensor->buff = CCML_BUFF_CNST;

    int size = ccml_size(tensor);
    tensor->data = malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        *((float*)tensor->data + i) = data[i];
    }
}

CCML_API void ccml_fill(ccml_tensor * tensor, float value) {
    tensor->type = CCML_TYPE_FP32;
    tensor->buff = CCML_BUFF_CNST;

    int size = ccml_size(tensor);
    tensor->data = malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        *((float*)tensor->data + i) = value;
    }
}

//
//  ███╗   ███╗██╗███████╗ ██████╗
//  ████╗ ████║██║██╔════╝██╔════╝
//  ██╔████╔██║██║███████╗██║
//  ██║╚██╔╝██║██║╚════██║██║
//  ██║ ╚═╝ ██║██║███████║╚██████╗
//  ╚═╝     ╚═╝╚═╝╚══════╝ ╚═════╝
//
//  ███████╗██╗   ██╗███╗   ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗███████╗
//  ██╔════╝██║   ██║████╗  ██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
//  █████╗  ██║   ██║██╔██╗ ██║██║        ██║   ██║██║   ██║██╔██╗ ██║███████╗
//  ██╔══╝  ██║   ██║██║╚██╗██║██║        ██║   ██║██║   ██║██║╚██╗██║╚════██║
//  ██║     ╚██████╔╝██║ ╚████║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║███████║
//  ╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
//

CCML_API bool ccml_can_broadcast(ccml_tensor * lhs, ccml_tensor * rhs) {
    if (rhs == NULL || lhs == NULL) return true;

    int v_lhs = lhs->view[0];
    int v_rhs = rhs->view[0];

    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        if (lhs->shape[v_lhs][i] != rhs->shape[v_rhs][i] &&
            lhs->shape[v_lhs][i] != 1 && rhs->shape[v_rhs][i] != 1) return false;
    }

    return true;
}

CCML_API bool ccml_is_indexable(ccml_tensor * tensor) {
    return (tensor->buff == CCML_BUFF_CNST && ccml_size(tensor) != 1) ||
           (tensor->buff == CCML_BUFF_LOAD || tensor->buff == CCML_BUFF_SAVE);
}

CCML_API bool ccml_is_leaf(ccml_tensor * tensor) {
    return tensor->src[0] == NULL && tensor->src[1] == NULL;
}

CCML_API bool ccml_has_buffer(ccml_tensor * tensor) {
    return tensor->buff != CCML_BUFF_NONE;
}

CCML_API int ccml_ndim(ccml_tensor * tensor) {
    int last_dim = 0;
    int v = tensor->view[0];
    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        if(tensor->shape[v][i] != 1) last_dim = i;
    }
    return last_dim == 0 ? 1 : last_dim + 1;
}

CCML_API bool ccml_is_vector(ccml_tensor * tensor) {
    int v = tensor->view[0];
    return tensor->shape[v][1] == 1 && tensor->shape[v][2] == 1 &&
           tensor->shape[v][3] == 1;
}

CCML_API bool ccml_is_matrix(ccml_tensor * tensor) {
    int v = tensor->view[0];
    return tensor->shape[v][0] != 1 && tensor->shape[v][1] != 1 &&
           tensor->shape[v][2] == 1 && tensor->shape[v][3] == 1;
}

//
//  ██████╗ ██████╗ ██╗███╗   ███╗ █████╗ ██████╗ ██╗   ██╗
//  ██╔══██╗██╔══██╗██║████╗ ████║██╔══██╗██╔══██╗╚██╗ ██╔╝
//  ██████╔╝██████╔╝██║██╔████╔██║███████║██████╔╝ ╚████╔╝
//  ██╔═══╝ ██╔══██╗██║██║╚██╔╝██║██╔══██║██╔══██╗  ╚██╔╝
//  ██║     ██║  ██║██║██║ ╚═╝ ██║██║  ██║██║  ██║   ██║
//  ╚═╝     ╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝
//
//   ██████╗ ██████╗ ███████╗██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗███████╗
//  ██╔═══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
//  ██║   ██║██████╔╝█████╗  ██████╔╝███████║   ██║   ██║██║   ██║██╔██╗ ██║███████╗
//  ██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗██╔══██║   ██║   ██║██║   ██║██║╚██╗██║╚════██║
//  ╚██████╔╝██║     ███████╗██║  ██║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║███████║
//   ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
//

CCML_API ccml_tensor * ccml_log(ccml_tensor * tensor) {
    ccml_tensor * result = ccml_new_tensor_impl(tensor->type, tensor->shape[tensor->view[0]]);

    result->oper = CCML_OPER_LOG;
    result->view[1] = tensor->view[0];
    result->src[0] = tensor;
    result->requires_grad = tensor->requires_grad;

    return result;
}

CCML_API ccml_tensor * ccml_exp(ccml_tensor * tensor) {
    ccml_tensor * result = ccml_new_tensor_impl(tensor->type, tensor->shape[tensor->view[0]]);

    result->oper = CCML_OPER_EXP;
    result->view[1] = tensor->view[0];
    result->src[0] = tensor;
    result->requires_grad = tensor->requires_grad;

    return result;
}

CCML_API ccml_tensor * ccml_sin(ccml_tensor * tensor) {
    ccml_tensor * result = ccml_new_tensor_impl(tensor->type, tensor->shape[tensor->view[0]]);

    result->oper = CCML_OPER_SIN;
    result->view[1] = tensor->view[0];
    result->src[0] = tensor;
    result->requires_grad = tensor->requires_grad;

    return result;
}

CCML_API ccml_tensor * ccml_rec(ccml_tensor * tensor) {
    ccml_tensor * result = ccml_new_tensor_impl(tensor->type, tensor->shape[tensor->view[0]]);

    result->oper = CCML_OPER_REC;
    result->view[1] = tensor->view[0];
    result->src[0] = tensor;
    result->requires_grad = tensor->requires_grad;

    return result;
}

CCML_API ccml_tensor * ccml_sqrt(ccml_tensor * tensor) {
    ccml_tensor * result = ccml_new_tensor_impl(tensor->type, tensor->shape[tensor->view[0]]);

    result->oper = CCML_OPER_SQRT;
    result->view[1] = tensor->view[0];
    result->src[0] = tensor;
    result->requires_grad = tensor->requires_grad;

    return result;
}

CCML_API ccml_tensor * ccml_add(ccml_tensor * lhs, ccml_tensor * rhs) {
    CCML_ASSERT(ccml_can_broadcast(lhs, rhs), "incompatible dimensions for broadcasting");

    bool null_input = lhs == NULL || rhs == NULL;

    int shape[CCML_DIMS_MAX] = {0};
    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        shape[i] = null_input ? lhs->shape[lhs->view[0]][i] : (lhs->shape[lhs->view[0]][i] +
            rhs->shape[rhs->view[0]][i] + abs(lhs->shape[lhs->view[0]][i] - rhs->shape[rhs->view[0]][i])) / 2;
    }

    ccml_tensor * result = ccml_new_tensor_impl(lhs->type, shape);

    result->oper = CCML_OPER_ADD;
    result->view[1] = lhs->view[0];
    result->view[2] = null_input ? 0 : rhs->view[0];
    result->src[0] = lhs;
    result->src[1] = rhs;
    result->requires_grad = null_input ? lhs->requires_grad : lhs->requires_grad || rhs->requires_grad;

    return result;
}

CCML_API ccml_tensor * ccml_mul(ccml_tensor * lhs, ccml_tensor * rhs) {
    CCML_ASSERT(ccml_can_broadcast(lhs, rhs), "incompatible dimensions for broadcasting");

    bool null_input = lhs == NULL || rhs == NULL;

    int shape[CCML_DIMS_MAX] = {0};
    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        shape[i] = null_input ? lhs->shape[lhs->view[0]][i] : (lhs->shape[lhs->view[0]][i] +
            rhs->shape[rhs->view[0]][i] + abs(lhs->shape[lhs->view[0]][i] - rhs->shape[rhs->view[0]][i])) / 2;
    }

    ccml_tensor * result = ccml_new_tensor_impl(lhs->type, shape);

    result->oper = CCML_OPER_MUL;
    result->view[1] = lhs->view[0];
    result->view[2] = null_input ? 0 : rhs->view[0];
    result->src[0] = lhs;
    result->src[1] = rhs;
    result->requires_grad = null_input ? lhs->requires_grad : lhs->requires_grad || rhs->requires_grad;

    return result;
}

CCML_API void ccml_reshape(ccml_tensor * tensor, int * shape) {
    int size = ccml_size(tensor);
    int new_size = shape[0] * shape[1] * shape[2] * shape[3];
    CCML_ASSERT(size == new_size, "reshaped and source tensor must have the same size");
    CCML_ASSERT(tensor->view[0] < CCML_VIEW_MAX - 1, "maximum number of views reached");

    int ne[CCML_DIMS_MAX] = {
        shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1
    };

    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        tensor->shape[tensor->view[0] + 1][i] = shape[i];
        tensor->stride[tensor->view[0] + 1][i] = ne[i];
    }

    tensor->view[0]++;
}

CCML_API void ccml_permute(ccml_tensor * tensor, int * perm) {
    CCML_ASSERT(tensor->view[0] < CCML_VIEW_MAX - 1, "maximum number of views reached");
    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        tensor->shape[tensor->view[0] + 1][i] = tensor->shape[tensor->view[0]][perm[i]];
        tensor->stride[tensor->view[0] + 1][i] = tensor->stride[tensor->view[0]][perm[i]];
    }

    tensor->view[0]++;
}

CCML_API ccml_tensor * ccml_sum(ccml_tensor * tensor, int n_axes, int axes[CCML_DIMS_MAX]) {
    CCML_ASSERT(n_axes >= 0 && n_axes <= CCML_DIMS_MAX, "invalid number of summed axes");

    int shape[CCML_DIMS_MAX] = {1, 1, 1, 1};
    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        shape[i] = tensor->shape[tensor->view[0]][i];
    }

    for (int i = 0; i < n_axes; i++) {
        shape[axes[i]] = 1;
    }

    ccml_tensor * result = ccml_new_tensor_impl(tensor->type, shape);

    result->oper = CCML_OPER_SUM;
    result->buff = CCML_BUFF_SAVE;
    result->view[1] = tensor->view[0];
    result->src[0] = tensor;
    result->requires_grad = tensor->requires_grad;

    return result;
}

//
//  ███████╗███████╗ ██████╗ ██████╗ ███╗   ██╗██████╗  █████╗ ██████╗ ██╗   ██╗
//  ██╔════╝██╔════╝██╔════╝██╔═══██╗████╗  ██║██╔══██╗██╔══██╗██╔══██╗╚██╗ ██╔╝
//  ███████╗█████╗  ██║     ██║   ██║██╔██╗ ██║██║  ██║███████║██████╔╝ ╚████╔╝
//  ╚════██║██╔══╝  ██║     ██║   ██║██║╚██╗██║██║  ██║██╔══██║██╔══██╗  ╚██╔╝
//  ███████║███████╗╚██████╗╚██████╔╝██║ ╚████║██████╔╝██║  ██║██║  ██║   ██║
//  ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝
//
//   ██████╗ ██████╗ ███████╗██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗███████╗
//  ██╔═══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
//  ██║   ██║██████╔╝█████╗  ██████╔╝███████║   ██║   ██║██║   ██║██╔██╗ ██║███████╗
//  ██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗██╔══██║   ██║   ██║██║   ██║██║╚██╗██║╚════██║
//  ╚██████╔╝██║     ███████╗██║  ██║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║███████║
//   ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
//

CCML_API ccml_tensor * ccml_neg(ccml_tensor * tensor) {
    return ccml_log(ccml_rec(ccml_exp(tensor)));
}

CCML_API ccml_tensor * ccml_sub(ccml_tensor * lhs, ccml_tensor * rhs) {
    return ccml_add(lhs, ccml_neg(rhs));
}

CCML_API ccml_tensor * ccml_div(ccml_tensor * lhs, ccml_tensor * rhs) {
    return ccml_mul(lhs, ccml_rec(rhs));
}

CCML_API ccml_tensor * ccml_square(ccml_tensor * tensor) {
    return ccml_mul(tensor, tensor);
}

CCML_API ccml_tensor * ccml_cos(ccml_tensor * tensor) {
    ccml_tensor * pi_2 = ccml_new_tensor_impl(tensor->type, (int[]){1, 1, 1, 1});
    ccml_fill(pi_2, M_PI_2);

    return ccml_sin(ccml_add(tensor, pi_2));
}

CCML_API ccml_tensor * ccml_tanh(ccml_tensor * tensor) {
    return ccml_div(ccml_sub(ccml_exp(tensor), ccml_exp(ccml_neg(tensor))),
                    ccml_add(ccml_exp(tensor), ccml_exp(ccml_neg(tensor))));
}

CCML_API ccml_tensor * ccml_matmul(ccml_tensor * lhs, ccml_tensor * rhs) {
    CCML_ASSERT(ccml_is_matrix(lhs));
    CCML_ASSERT(ccml_is_matrix(rhs));

    CCML_ASSERT(lhs->shape[lhs->view[0]][1] == rhs->shape[rhs->view[0]][0]);

    ccml_reshape(lhs, (int[]){lhs->shape[lhs->view[0]][0], lhs->shape[lhs->view[0]][1], 1, 1});
    ccml_reshape(rhs, (int[]){1, rhs->shape[rhs->view[0]][0], rhs->shape[rhs->view[0]][1], 1});

    ccml_tensor * mul = ccml_mul(lhs, rhs);
    ccml_tensor * sum = ccml_sum(mul, 1, (int[]){1});

    ccml_reshape(sum, (int[]){sum->shape[sum->view[0]][0], sum->shape[sum->view[0]][2], 1, 1});

    return sum;
}

CCML_API ccml_tensor * ccml_soft_max(ccml_tensor * tensor) {
    return ccml_div(ccml_exp(tensor), ccml_sum(ccml_exp(tensor), 4, (int[]){0, 1, 2, 3}));
}

CCML_API ccml_tensor * ccml_cross_entropy_loss(ccml_tensor * tensor, ccml_tensor * target) {
    CCML_ASSERT(ccml_size(tensor) == ccml_size(target));

    return ccml_neg(ccml_sum(ccml_mul(target, ccml_log(tensor)), 4, (int[]){0, 1, 2, 3}));
}

//
//  ██╗  ██╗ █████╗ ███████╗██╗  ██╗███╗   ███╗ █████╗ ██████╗
//  ██║  ██║██╔══██╗██╔════╝██║  ██║████╗ ████║██╔══██╗██╔══██╗
//  ███████║███████║███████╗███████║██╔████╔██║███████║██████╔╝
//  ██╔══██║██╔══██║╚════██║██╔══██║██║╚██╔╝██║██╔══██║██╔═══╝
//  ██║  ██║██║  ██║███████║██║  ██║██║ ╚═╝ ██║██║  ██║██║
//  ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝
//

#define CCML_FNV_PRIME 1099511628211LU
#define CCML_FNV_OFFSET 14695981039346656037LU

typedef struct ccml_hashmap_entry {
    uintptr_t key;
    int value;
} ccml_hashmap_entry;

typedef struct ccml_hashmap {
    int used;
    ccml_hashmap_entry entries[CCML_NODE_MAX];
    int capacity;
} ccml_hashmap;

CCML_API uint64_t ccml_hash_key(void * key) {
    uint64_t hash = CCML_FNV_OFFSET;
    hash ^= (uint64_t)(uintptr_t)key;
    hash *= CCML_FNV_PRIME;
    return hash;
}

CCML_API ccml_hashmap * ccml_new_hashmap() {
    int capacity = CCML_NODE_MAX;

    ccml_hashmap * map = malloc(sizeof(ccml_hashmap));
    *map = (ccml_hashmap){
        .used = 0,
        .entries = {0},
        .capacity = capacity,
    };

    for (int i = 0; i < capacity; i++) {
        map->entries[i].key = 0;
        map->entries[i].value = -1;
    }

    return map;
}

CCML_API int ccml_hashmap_get(ccml_hashmap * map, void * key) {
    if (key == NULL) {
        return -1;
    }

    uint64_t hash = ccml_hash_key(key);
    int index = (int)(hash & (uint64_t)(map->capacity - 1));

    while (map->entries[index].key != 0) {
        if ((uintptr_t)key == map->entries[index].key) {
            return map->entries[index].value;
        }

        index++;
        if (index >= map->capacity) {
            index = 0;
        }
    }

    return -1;
};

CCML_API void ccml_hashmap_set(ccml_hashmap * map, void * key, int value) {
    if (map->used >= map->capacity) {
        CCML_ASSERT(false, "hashmap size overflow");
    }

    uint64_t hash = ccml_hash_key(key);
    int index = (int)(hash & (int)(map->capacity - 1));

    while (map->entries[index].key != 0) {
        if ((uintptr_t)key == map->entries[index].key) {
            // Found key (it already exists), update value.
            map->entries[index].value = value;
            return;
        }
        // Key wasn't in this slot, move to next (linear
        // probing).
        index++;
        if (index >= map->capacity) {
            index = 0;
        }
    }

    map->entries[index].key = (uintptr_t)key;
    map->entries[index].value = value;
    map->used++;
}

//
//   ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗
//  ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║
//  ██║  ███╗██████╔╝███████║██████╔╝███████║
//  ██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║
//  ╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║
//   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
//

typedef struct ccml_graph {
    int n_nodes;
    ccml_tensor * nodes[CCML_NODE_MAX];
    ccml_hashmap * map;
} ccml_graph;

CCML_API void
ccml_graph_forward(struct ccml_graph * graph, ccml_tensor * tensor, int * node_counter) {
    if (tensor == NULL) return;

    // also checking if tensor has itself as a child to prevent (infinite)
    // cycles

    if (ccml_hashmap_get(graph->map, tensor->src[0]) == -1) {
        ccml_graph_forward(graph, tensor->src[0], node_counter);
    }
    if (ccml_hashmap_get(graph->map, tensor->src[1]) == -1) {
        ccml_graph_forward(graph, tensor->src[1], node_counter);
    }

    if (ccml_hashmap_get(graph->map, tensor) == -1) {
        CCML_ASSERT(*node_counter < CCML_NODE_MAX - 1, "more nodes created than CCML_NODE_MAX");
        tensor->index = *node_counter;
        graph->nodes[*node_counter] = tensor;
        ccml_hashmap_set(graph->map, tensor, (*node_counter)++);
    }
}

CCML_API void ccml_graph_backward(ccml_graph * graph, ccml_tensor * root) {
    if (root->requires_grad == false) return;
    ccml_tensor * one = ccml_new_tensor_impl(root->type, root->shape[root->view[0]]);
    ccml_fill(one, 1.0f);
    root->grad = ccml_mul(root, one);

    // in this loop create gradient tensors corresponding to each tensor
    // that requires gradient tracking, and set their buffers to the correct
    // option respectively (because, for example, a intermediary tensor w/o
    // a buffer also needs an intermediary gradient tensor w/o a buffer)

    ccml_tensor * queue[CCML_NODE_MAX] = {NULL};
    int queue_start = 0;
    int queue_end = 0;
    queue[queue_end++] = root;

    while (queue_end != queue_start) {
        ccml_tensor * tensor = queue[queue_start++];
        if (tensor->requires_grad == true) {
            // declaring partials d(tensor)/d(tensor->src[0]) and
            // d(tensor)/d(tensor->src[1])

            ccml_tensor * one = ccml_new_tensor_impl(tensor->type, (int[]){1, 1, 1, 1});
            ccml_tensor * two = ccml_new_tensor_impl(tensor->type, (int[]){1, 1, 1, 1});

            ccml_fill(one, 1.0f);
            ccml_fill(two, 2.0f);

            ccml_tensor * partial_0 = one;
            ccml_tensor * partial_1 = one;

            // calculating partials

            switch (tensor->oper) {
                case CCML_OPER_NONE:
                    partial_0 = one; break;
                case CCML_OPER_LOG:
                    partial_0 = ccml_rec(tensor->src[0]); break;
                case CCML_OPER_EXP:
                    partial_0 = ccml_exp(tensor->src[0]); break;
                case CCML_OPER_SIN:
                    partial_0 = ccml_cos(tensor->src[0]); break;
                case CCML_OPER_REC:
                    partial_0 = ccml_neg(ccml_rec(ccml_square(tensor->src[0]))); break;
                case CCML_OPER_SQRT:
                    partial_0 = ccml_rec(ccml_mul(two, ccml_sqrt(tensor->src[0]))); break;
                case CCML_OPER_ADD:
                    partial_0 = one; partial_1 = one; break;
                case CCML_OPER_MUL:
                    partial_0 = tensor->src[1]; partial_1 = tensor->src[0]; break;
                case CCML_OPER_SUM:
                    partial_0 = one; break;
                default:
                    CCML_ASSERT(false, "unknown variant of ccml_oper");
            }

            // multiplying tensor->grad by partials and adding them to
            // the gradients of the tensor's children (we have to do a
            // mini DFS traversal w/ ccml_graph_forward() since the gradient
            // calculation forms a mini sub-graph that needs to be tra-
            // versed separately)

            if (tensor->src[0] != NULL) {
                if (ccml_is_leaf(tensor->src[0])) {
                    tensor->src[0]->grad = ccml_add(ccml_mul(tensor->grad, partial_0), tensor->src[0]->grad);
                    ccml_fill(tensor->src[0]->grad, 0.0f);
                } else {
                    tensor->src[0]->grad = ccml_mul(ccml_mul(tensor->grad, partial_0), tensor->src[0]->grad);
                    ccml_fill(tensor->src[0]->grad, 1.0f);
                }
                ccml_graph_forward(graph, tensor->src[0]->grad, &graph->n_nodes);
                queue[queue_end++] = tensor->src[0];
            }
            if (tensor->src[1] != NULL) {
                if (ccml_is_leaf(tensor->src[1])) {
                    tensor->src[1]->grad = ccml_add(ccml_mul(tensor->grad, partial_1), tensor->src[1]->grad);
                    ccml_fill(tensor->src[1]->grad, 0.0f);
                } else {
                    tensor->src[1]->grad = ccml_mul(ccml_mul(tensor->grad, partial_1), tensor->src[1]->grad);
                    ccml_fill(tensor->src[1]->grad, 1.0f);
                }
                ccml_graph_forward(graph, tensor->src[1]->grad, &graph->n_nodes);
                queue[queue_end++] = tensor->src[1];
            }
        }
    }
}

CCML_API void ccml_graph_allocate_nodes(ccml_graph * graph) {
    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        int size = ccml_size(tensor);
        if (ccml_has_buffer(tensor) && tensor->data == NULL) {
            tensor->data = malloc(size * ccml_type_sizes[tensor->type]);
        }
        if ((tensor->buff == CCML_BUFF_LOAD || tensor->buff == CCML_BUFF_SAVE) && tensor->grad != NULL) {
            tensor->grad->buff = CCML_BUFF_SAVE;
            tensor->grad->data = malloc(size * ccml_type_sizes[tensor->type]);
        }
    }
}

CCML_API void ccml_graph_cse(ccml_graph * graph) {
    // an expression takes the following format of an integer
    // (oper->int) * 10^6 + (src0->index) * 10^3 + (src1->index)
    // with absent/NULL children being assigned index CCML_NODE_MAX

    CCML_ASSERT(CCML_NODE_MAX < 999, "ccml_graph_cse must be adjusted w/ the increased node count");

    // erasing graph->map hashmap
    for (int i = 0; i < graph->map->capacity; i++) {
        graph->map->used = 0;
        graph->map->entries[i].key = 0;
        graph->map->entries[i].value = -1;
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        if (ccml_is_leaf(tensor) == false) {
            int operation = tensor->oper;
            int src_0 = tensor->src[0] == NULL ? CCML_NODE_MAX : tensor->src[0]->index;
            int src_1 = tensor->src[1] == NULL ? CCML_NODE_MAX : tensor->src[1]->index;
            int expression = operation * pow(10, 6) + src_0 * pow(10, 3) + src_1;

            if (ccml_hashmap_get(graph->map, (void *)(uintptr_t)expression) == -1) {
                ccml_hashmap_set(graph->map, (void *)(uintptr_t)expression, tensor->index);
            } else {
                int index = ccml_hashmap_get(graph->map, (void *)(uintptr_t)expression);
                tensor->src[0] = graph->nodes[index];
                tensor->src[1] = NULL;
                tensor->oper = CCML_OPER_NONE;
            }
        }
    }
}

CCML_API ccml_graph * ccml_new_graph(ccml_tensor * root) {
    struct ccml_graph * graph = malloc(sizeof(struct ccml_graph));
    root->buff = CCML_BUFF_SAVE;

    *graph = (struct ccml_graph){
        /*.n_nodes =*/ 0,
        /*.nodes   =*/ {NULL},
        /*.map     =*/ ccml_new_hashmap()
    };

    ccml_graph_forward(graph, root, &graph->n_nodes);
    ccml_graph_backward(graph, root);

    ccml_graph_cse(graph);
    ccml_graph_allocate_nodes(graph);

    return graph;
};

CCML_API void ccml_graph_free(ccml_graph * graph) {
    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        // only freeing when tensor isn't of type reshape/permute, because those tensors
        // just use their children's data pointer, so we avoid a double free this way :)
        free(tensor->data);
        free(tensor);
    }
    free(graph->map);
    free(graph);
}

//
//  ██╗███╗   ██╗██████╗ ██╗ ██████╗███████╗███████╗
//  ██║████╗  ██║██╔══██╗██║██╔════╝██╔════╝██╔════╝
//  ██║██╔██╗ ██║██║  ██║██║██║     █████╗  ███████╗
//  ██║██║╚██╗██║██║  ██║██║██║     ██╔══╝  ╚════██║
//  ██║██║ ╚████║██████╔╝██║╚██████╗███████╗███████║
//  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝ ╚═════╝╚══════╝╚══════╝
//

typedef enum ccml_index {
    // index generation for a regular a[x][y][z] = x * (y * z) + y * (z) + z * (1) index
    CCML_INDEX_NORMAL,
    // index generation for a binary operation (left hand side)
    CCML_INDEX_BINARY_LHS,
    // index generation for a binary operation (right hand side)
    CCML_INDEX_BINARY_RHS,
    // index generation for a reduce operation (for the child)
    CCML_INDEX_REDUCE
} ccml_index;

CCML_API const char * ccml_new_index(ccml_tensor * tensor, ccml_index index) {
    int size = CCML_CHAR_MAX * CCML_DIMS_MAX;
    char * result = malloc(size * sizeof(char));
    *result = '\0';
    strncat(result, "[", size);

    bool condition = false;
    // operand is a tensor for whom index generation is actually performed
    ccml_tensor * operand = tensor;

    switch(index) {
        case CCML_INDEX_NORMAL:
            for (int i = 0; i < ccml_ndim(tensor); i++) {
                condition = ccml_size(tensor) != 1;
                snprintf(result + strlen(result), size - strlen(result), "%sid%d*%d*%d",
                         i != 0 && i != ccml_ndim(tensor) ? "+" : "", i,
                         tensor->stride[tensor->view[0]][i], condition ? 1 : 0);
            }
            operand = tensor;
            break;
        case CCML_INDEX_BINARY_LHS:
            for (int i = 0; i < ccml_ndim(tensor->src[0]); i++) {
                condition = tensor->src[0]->shape[tensor->src[0]->view[0]][i] != 1;
                snprintf(result + strlen(result), size - strlen(result), "%sid%d*%d*%d",
                         i != 0 && i != ccml_ndim(tensor->src[0]) ? "+" : "", i,
                         tensor->src[0]->stride[tensor->view[1]][i], condition ? 1 : 0);
            }
            operand = tensor->src[0];
            break;
        case CCML_INDEX_BINARY_RHS:
            for (int i = 0; i < ccml_ndim(tensor->src[1]); i++) {
                condition = tensor->src[1]->shape[tensor->src[1]->view[0]][i] != 1;
                snprintf(result + strlen(result), size - strlen(result), "%sid%d*%d*%d",
                         i != 0 && i != ccml_ndim(tensor->src[1]) ? "+" : "", i,
                         tensor->src[1]->stride[tensor->view[2]][i], condition ? 1 : 0);
            }
            operand = tensor->src[1];
            break;
        case CCML_INDEX_REDUCE:
            for (int i = 0; i < ccml_ndim(tensor->src[0]); i++) {
                condition = tensor->shape[tensor->view[1]][i] != 1;
                snprintf(result + strlen(result), size - strlen(result), "%sid%d*%d*%d",
                         i != 0 && i != ccml_ndim(tensor->src[0]) ? "+" : "", i,
                         tensor->stride[tensor->view[1]][i], condition ? 1 : 0);
            }
            operand = tensor;
            break;
        default: CCML_ASSERT(false, "unknown variant of ccml_index");
    }

    strncat(result + strlen(result), "]", size - strlen(result));
    if (ccml_is_indexable(operand) == false) {
        snprintf(result, size, "");
    }

    return result;
}

void ccml_kernel_indices(ccml_graph * graph, int * n_kernels, int kernels[CCML_KERN_MAX][2]) {
    kernels[*n_kernels][0] = 0;

    // not sure yet what's the optimal strategy for kernel slicing,
    // therefore for now everything is fused into a single kernel

    kernels[*n_kernels][1] = graph->n_nodes;
    *n_kernels += 1;
}

//
//  ██████╗  █████╗  ██████╗██╗  ██╗███████╗███╗   ██╗██████╗
//  ██╔══██╗██╔══██╗██╔════╝██║ ██╔╝██╔════╝████╗  ██║██╔══██╗
//  ██████╔╝███████║██║     █████╔╝ █████╗  ██╔██╗ ██║██║  ██║
//  ██╔══██╗██╔══██║██║     ██╔═██╗ ██╔══╝  ██║╚██╗██║██║  ██║
//  ██████╔╝██║  ██║╚██████╗██║  ██╗███████╗██║ ╚████║██████╔╝
//  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═════╝
//
//  ███╗   ███╗███████╗████████╗ █████╗ ██╗
//  ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║
//  ██╔████╔██║█████╗     ██║   ███████║██║
//  ██║╚██╔╝██║██╔══╝     ██║   ██╔══██║██║
//  ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║███████╗
//  ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝
//

CCML_API const char * ccml_oper_metal(ccml_oper oper) {
    switch (oper) {
        case CCML_OPER_LOG:  return "log";
        case CCML_OPER_EXP:  return "exp";
        case CCML_OPER_SIN:  return "sin";
        case CCML_OPER_REC:  return "1/";
        case CCML_OPER_SQRT: return "sqrt";
        case CCML_OPER_ADD:  return "+";
        case CCML_OPER_MUL:  return "*";
        default: CCML_ASSERT(false, "no meaningful conversion to string exists");
    }
}

CCML_API const char * ccml_type_metal(ccml_type type) {
    switch (type) {
        case CCML_TYPE_FP32: return "float ";
        default: CCML_ASSERT(false, "unknown variant of ccml_type");
    }
}

CCML_API const char * ccml_type_swift(ccml_type type) {
    switch(type) {
        case CCML_TYPE_FP32: return "Float";
        default: CCML_ASSERT(false, "unknown variant of ccml_type");
    }
}

CCML_API const char * ccml_kernel_metal(struct ccml_graph * graph, int n_kernel, int start, int finish) {
    int size = CCML_NODE_MAX * CCML_CHAR_MAX;
    char * kernel = malloc(size * sizeof(char));
    char * string = kernel;
    *string = '\0';

    // the n_kernel parameter specifies the index of the kernel being generated, and start to finish
    // are indexes of graph->nodes that pertain to that kernel

    if (n_kernel == 0) {
        string += snprintf(string, size - (kernel - string), "#include <metal_stdlib>\nusing namespace metal;\n");
    }

    string += snprintf(string, size - (kernel - string), "kernel void my_kernel_%d(", n_kernel);

    // adding kernel input parameters to the kernel string
    int n_kernel_parameters = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];

        if (ccml_is_indexable(tensor)) {
            if (tensor->oper == CCML_OPER_SUM) {}
            if (n_kernel_parameters == 0) {
                string += snprintf(string, size - (kernel - string), "device %s* data_%d [[buffer(0)]]",
                                   ccml_type_metal(tensor->type), i);
                n_kernel_parameters++;
            } else {
                string += snprintf(string, size - (kernel - string), ", device %s* data_%d [[buffer(%d)]]",
                                   ccml_type_metal(tensor->type), i, n_kernel_parameters);
                n_kernel_parameters++;
            }
        }
    }

   int grid[CCML_DIMS_MAX] = {1, 1, 1, 1};
   for (int i = start; i < finish; i++) {
       ccml_tensor * tensor = graph->nodes[i];
       int v = tensor->view[0];

       grid[0] = grid[0] > tensor->shape[v][0] ? grid[0] : tensor->shape[v][0];
       grid[1] = grid[1] > tensor->shape[v][1] ? grid[1] : tensor->shape[v][1];
   }

    string += snprintf(string, size - (kernel - string),
                       ", uint3 gid [[thread_position_in_grid]]) {\n"
                       "\tuint id0 = gid.x / %d;\n\tuint id1 = gid.x %% %d;\n"
                       "\tuint id2 = gid.y;\n\tuint id3 = gid.z;\n\n",
                       grid[1], grid[1]);

    for (int i = start; i < finish; i++) {
        ccml_tensor * tensor = graph->nodes[i];

        switch (tensor->oper) {
            case CCML_OPER_NONE:
                if (ccml_is_leaf(tensor) == false) {
                    string += snprintf(string, size - (kernel - string), "\t%s%s data_%d%s = data_%d%s;\n",
                                       ccml_type_metal(tensor->type), ccml_is_indexable(tensor) ? "*" : "",
                                       tensor->index, ccml_new_index(tensor, CCML_INDEX_NORMAL), tensor->src[0]->index,
                                       ccml_new_index(tensor->src[0], CCML_INDEX_NORMAL));
                }
                // tensor data is embeddeable directly into the kernel string
                if (ccml_has_buffer(tensor) && ccml_is_indexable(tensor) == false) {
                    string += snprintf(string, size - (kernel - string), "\t%sdata_%d = %f;\n",
                                       ccml_type_metal(tensor->type), i, *(float *)tensor->data);
                }
                break;
            case CCML_OPER_LOG:
            case CCML_OPER_EXP:
            case CCML_OPER_SIN:
            case CCML_OPER_REC:
            case CCML_OPER_SQRT:
                string += snprintf(string, size - (kernel - string), "\t%sdata_%d%s = ",
                                   ccml_is_indexable(tensor) ? "" : ccml_type_metal(tensor->type),
                                   tensor->index, ccml_new_index(tensor, CCML_INDEX_NORMAL));
                string += snprintf(string, size - (kernel - string), "%s(data_%d%s);\n",
                                   ccml_oper_metal(tensor->oper),
                                   tensor->src[0]->index, ccml_new_index(tensor->src[0], CCML_INDEX_NORMAL));
                break;
            case CCML_OPER_ADD:
            case CCML_OPER_MUL:
                string += snprintf(string, size - (kernel - string), "\t%sdata_%d%s = ",
                                   ccml_is_indexable(tensor) ? "" : ccml_type_metal(tensor->type),
                                   tensor->index, ccml_new_index(tensor, CCML_INDEX_NORMAL));
                string += snprintf(string, size - (kernel - string), "data_%d%s %s ",
                                   tensor->src[0]->index, ccml_new_index(tensor, CCML_INDEX_BINARY_LHS),
                                   ccml_oper_metal(tensor->oper));
                string += snprintf(string, size - (kernel - string), "data_%d%s;\n",
                                   tensor->src[1] != NULL ? tensor->src[1]->index : tensor->index,
                                   tensor->src[1] != NULL ? ccml_new_index(tensor, CCML_INDEX_BINARY_RHS) :
                                   ccml_new_index(tensor, CCML_INDEX_NORMAL));
                break;
            case CCML_OPER_SUM:
                string += snprintf(string, size - (kernel - string),
                                   "\tfor (int i = 0; i < %d; i++) {\n\t\tdata_%d%s += ",
                                   ccml_size(tensor->src[0])/ccml_size(tensor),
                                   tensor->index, ccml_new_index(tensor, CCML_INDEX_REDUCE));
                string += snprintf(string, size - (kernel - string), "data_%d%s;\n\t}\n",
                                   tensor->src[0]->index,
                                   ccml_new_index(tensor->src[0], CCML_INDEX_NORMAL));
                break;
            default: CCML_ASSERT(false, "unknown variant of ccml_oper");
        }
    }

    string += snprintf(string, size - (kernel - string), "}");
    return kernel;
}

CCML_API void ccml_code_metal(ccml_graph * graph) {
    FILE * file_ptr = fopen("metal.swift", "w");
    CCML_ASSERT(file_ptr != NULL, "failed to create file for metal source");

    FILE * file_kernel_ptr = fopen("kernel.metal", "w");
    CCML_ASSERT(file_kernel_ptr != NULL, "failed to create a file for kernel source");

    // kernels[i][0] and kernels[i][1] specifies the starting and finishing indexes
    // of graph->nodes that pertain to i-th kernel
    int kernels[CCML_KERN_MAX][2] = {0};
    int n_kernels = 0;
    ccml_kernel_indices(graph, &n_kernels, kernels);

    // general metal setup
    fprintf(file_ptr, "import MetalKit\n\n"
                      "let device = MTLCreateSystemDefaultDevice()!\n"
                      "let filePath = Bundle.main.path(forResource: \"kernel\", ofType: \"metal\")!\n"
                      "let source = try! String(contentsOfFile: filePath, encoding: .utf8)\n"
                      "let library = try! device.makeLibrary(source: source, options: nil)\n"
                      "let commandQueue = device.makeCommandQueue()!\n\n");

    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        if (ccml_is_indexable(tensor)) {
            fprintf(file_ptr, "var cpu_buffer_%d: [%s] = [%f", i, ccml_type_swift(tensor->type), tensor->data[0]);
            for (int j = 1; j < ccml_size(tensor); j++) {
                fprintf(file_ptr, ", %f", tensor->data[j]);
            }
            fprintf(file_ptr, "]\n");
            fprintf(file_ptr, "let gpu_buffer_%d = device.makeBuffer(bytes: cpu_buffer_%d, length: %d * MemoryLayout<%s>.size, options: [])\n",
                              i, i, ccml_size(tensor), ccml_type_swift(tensor->type));
        }
    }

    // code specific to each i-th kernel
    for (int i = 0; i < n_kernels; i++) {
        fprintf(file_kernel_ptr, "%s\n\n", ccml_kernel_metal(graph, i, kernels[i][0], kernels[i][1]));

        fprintf(file_ptr, "\n// setting up my_kernel_%d\n"
                          "let kernelFunction_%d = library.makeFunction(name: \"my_kernel_%d\")!\n"
                          "let computePipeline_%d = try! device.makeComputePipelineState(function: kernelFunction_%d)\n"
                          "let commandBuffer_%d = commandQueue.makeCommandBuffer()!\n"
                          "let computeEncoder_%d = commandBuffer_%d.makeComputeCommandEncoder()!\n"
                          "computeEncoder_%d.setComputePipelineState(computePipeline_%d)\n\n",
                          i, i, i, i, i, i, i, i, i,i );

        // code specific to i-th kernel's buffers
        int thread_grid[CCML_DIMS_MAX] = {1, 1, 1, 1};
        int n_buffers = 0;
        for (int j = 0; j < graph->n_nodes; j++) {
            ccml_tensor * tensor = graph->nodes[j];
            if (ccml_is_indexable(tensor)) {
                int v = tensor->view[0];

                if (j >= kernels[i][0] && j <= kernels[i][1]) {
                    thread_grid[0] = thread_grid[0] > tensor->shape[v][0] ? thread_grid[0] : tensor->shape[v][0];
                    thread_grid[1] = thread_grid[1] > tensor->shape[v][1] ? thread_grid[1] : tensor->shape[v][1];
                    thread_grid[2] = thread_grid[2] > tensor->shape[v][2] ? thread_grid[2] : tensor->shape[v][2];
                    thread_grid[3] = thread_grid[3] > tensor->shape[v][3] ? thread_grid[3] : tensor->shape[v][3];
                }

                fprintf(file_ptr, "computeEncoder_%d.setBuffer(gpu_buffer_%d, offset: 0, index: %d)\n",
                                   i, j, n_buffers++);
            }
        }

        fprintf(file_ptr, "\nlet gridSize_%d = MTLSize(width: %d, height: %d, depth: %d)\n"
                          "let threadGroupSize_%d = MTLSize(width: 1, height: 1, depth: 1)\n"
                          "computeEncoder_%d.dispatchThreads(gridSize_%d, threadsPerThreadgroup: threadGroupSize_%d)\n"
                          "computeEncoder_%d.endEncoding()\n"
                          "commandBuffer_%d.commit()\n"
                          "commandBuffer_%d.waitUntilCompleted()\n\n",
                          i, thread_grid[0] * thread_grid[1], thread_grid[2], thread_grid[3],
                          i, i, i, i, i, i, i);

        // copying data back from GPU to buffers (specific to i-th kernel)
        for (int j = kernels[i][0]; j < kernels[i][1]; j++) {
            ccml_tensor * tensor = graph->nodes[j];
            if (ccml_is_indexable(tensor)) {
                fprintf(file_ptr, "memcpy(&cpu_buffer_%d, gpu_buffer_%d!.contents(), cpu_buffer_%d.count * MemoryLayout<%s>.size);\n"
                                  "print(cpu_buffer_%d)\n", j, j, j, ccml_type_swift(tensor->type), j);

            }
        }
    }

    #if defined(__APPLE__)
        system("swiftc -o metal metal.swift -framework MetalKit && ./metal");
    #else
        #warning "platform not supported, code generated"
    #endif
}



//
//  ███████╗██╗  ██╗███████╗ ██████╗██╗   ██╗████████╗██╗ ██████╗ ███╗   ██╗
//  ██╔════╝╚██╗██╔╝██╔════╝██╔════╝██║   ██║╚══██╔══╝██║██╔═══██╗████╗  ██║
//  █████╗   ╚███╔╝ █████╗  ██║     ██║   ██║   ██║   ██║██║   ██║██╔██╗ ██║
//  ██╔══╝   ██╔██╗ ██╔══╝  ██║     ██║   ██║   ██║   ██║██║   ██║██║╚██╗██║
//  ███████╗██╔╝ ██╗███████╗╚██████╗╚██████╔╝   ██║   ██║╚██████╔╝██║ ╚████║
//  ╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝    ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
//

CCML_API void ccml_graph_execute(ccml_graph * graph) {
    ccml_code_metal(graph);
}

#endif /* CCML_IMPL */
