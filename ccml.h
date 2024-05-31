#if !defined(CCML_IMPL)
#define CCML_IMPL

#define __STDC_WANT_IEC_60559_TYPES_EXT__
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdalign.h>

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199409L) && !defined(CCML_API)
    #define CCML_API static inline
#elif !defined(CCML_API)
    #define CCML_API static
#endif

#define CCML_ASSERT(x, ...) do { if (!(x)) {                                     \
    fprintf(stderr, "CCML_ASSERT: %s:%d: %s", __FILE__, __LINE__, #x);           \
    __VA_OPT__(fprintf(stderr, ", %s", __VA_ARGS__);)                            \
    exit(EXIT_FAILURE);                                                          \
} } while (0)

#define CCML_SRCS_MAX 2
#define CCML_TYPE_MAX 3
#define CCML_DIMS_MAX 4

#define CCML_KERN_MAX 16
#define CCML_CHAR_MAX 80
#define CCML_NODE_MAX 128

// known issues:
// - including ccml.h in separate compilation units compiles separate/independent symbols
// - a lot of function return statuses aren't checked, mostly snprintf/fread/fwrite

//
//   ██████╗ ██████╗ ███╗   ██╗████████╗███████╗██╗  ██╗████████╗
//  ██╔════╝██╔═══██╗████╗  ██║╚══██╔══╝██╔════╝╚██╗██╔╝╚══██╔══╝
//  ██║     ██║   ██║██╔██╗ ██║   ██║   █████╗   ╚███╔╝    ██║
//  ██║     ██║   ██║██║╚██╗██║   ██║   ██╔══╝   ██╔██╗    ██║
//  ╚██████╗╚██████╔╝██║ ╚████║   ██║   ███████╗██╔╝ ██╗   ██║
//   ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝
//

typedef struct ccml_context {
    int capacity;
    int used;
    void * memory;
} ccml_context;

CCML_API ccml_context * ccml_new_context(void * memory, int capacity) {
    CCML_ASSERT(memory != NULL && capacity > 0);
    int max_align = alignof(max_align_t);
    ccml_context * ctx = memory;

    *ctx = (ccml_context) {
        /*.capacity =*/ capacity,
        /*.used     =*/ (sizeof(ccml_context)/max_align + 1) * max_align,
        /*.memory   =*/ memory
    };

    return ctx;
}

CCML_API void * ccml_malloc(ccml_context * ctx, int size) {
    int max_align = alignof(max_align_t);
    int size_aligned = (size/max_align + 1) * max_align;

    CCML_ASSERT(ctx->used + size_aligned < ctx->capacity);
    void * ptr = ctx->memory + ctx->used;
    ctx->used += size_aligned;

    return ptr;
}

CCML_API void ccml_context_free(ccml_context * ctx) {
    free(ctx->memory);
}

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
    CCML_BUFF_NEW,
    // dedicated buffer for constant values
    CCML_BUFF_CNST,
    // dedicated buffer for loaded values
    CCML_BUFF_LOAD,
    // dedicated buffer for saved values
    CCML_BUFF_SAVE
} ccml_buff;

typedef enum ccml_oper {
    CCML_OPER_NEW,
    CCML_OPER_COPY,
    CCML_OPER_LOG,
    CCML_OPER_EXP,
    CCML_OPER_SIN,
    CCML_OPER_REC,
    CCML_OPER_SQRT,
    CCML_OPER_ADD,
    CCML_OPER_MUL,
    CCML_OPER_SUM,
    CCML_OPER_RES,
    CCML_OPER_PER
} ccml_oper;

typedef struct ccml_tensor {
    enum ccml_type type;
    enum ccml_buff buff;
    enum ccml_oper oper;

    int shape[CCML_DIMS_MAX];
    int stride[CCML_DIMS_MAX];

    struct ccml_tensor * src[CCML_SRCS_MAX];
    struct ccml_tensor * grad;

    void * data;

    int index;
    bool is_grad;
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

CCML_API ccml_tensor * ccml_new_tensor_impl(ccml_context * ctx, ccml_type type,
    int * shape, ccml_tensor * lhs, ccml_tensor * rhs, bool is_grad
) {
    ccml_tensor * result = ccml_malloc(ctx, sizeof(ccml_tensor));
    int ne[CCML_DIMS_MAX] = {
        shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1
    };

    *result = (ccml_tensor) {
        /*.type    =*/ type,
        /*.buff    =*/ CCML_BUFF_NEW,
        /*.oper    =*/ CCML_OPER_NEW,
        /*.shape   =*/ {shape[0], shape[1], shape[2], shape[3]},
        /*.stride  =*/ {ne[0], ne[1], ne[2], ne[3]},
        /*.src     =*/ {lhs, rhs},
        /*.grad    =*/ NULL,
        /*.data    =*/ NULL,
        /*.index   =*/ -1,
        /*.is_grad =*/ is_grad,
    };

    return result;
}

CCML_API ccml_tensor * ccml_new_tensor_1d(ccml_context * ctx,
    ccml_type type, int ne0, bool is_grad
) {
    int shape[CCML_DIMS_MAX] = {ne0, 1, 1, 1};

    ccml_tensor * result = ccml_new_tensor_impl(ctx, type, shape, NULL, NULL, is_grad);
    result->buff = CCML_BUFF_LOAD;

    return result;
}

CCML_API ccml_tensor * ccml_new_tensor_2d(ccml_context * ctx,
    ccml_type type, int ne0, int ne1, bool is_grad
) {
    int shape[CCML_DIMS_MAX] = {ne0, ne1, 1, 1};

    ccml_tensor * result = ccml_new_tensor_impl(ctx, type, shape, NULL, NULL, is_grad);
    result->buff = CCML_BUFF_LOAD;

    return result;
}

CCML_API ccml_tensor * ccml_new_tensor_3d(ccml_context * ctx,
    ccml_type type, int ne0, int ne1, int ne2, bool is_grad
) {
    int shape[CCML_DIMS_MAX] = {ne0, ne1, ne2, 1};

    ccml_tensor * result = ccml_new_tensor_impl(ctx, type, shape, NULL, NULL, is_grad);
    result->buff = CCML_BUFF_LOAD;

    return result;
}

CCML_API ccml_tensor * ccml_new_tensor_4d(ccml_context * ctx,
    ccml_type type, int ne0, int ne1, int ne2, int ne3, bool is_grad
) {
    int shape[CCML_DIMS_MAX] = {ne0, ne1, ne2, ne3};

    ccml_tensor * result = ccml_new_tensor_impl(ctx, type, shape, NULL, NULL, is_grad);
    result->buff = CCML_BUFF_LOAD;

    return result;
}

CCML_API int ccml_size(ccml_tensor * tensor) {
    return tensor->shape[0] * tensor->shape[1] * tensor->shape[2] * tensor->shape[3];
}

CCML_API void ccml_set(ccml_context * ctx, ccml_tensor * tensor, float * data) {
    tensor->type = CCML_TYPE_FP32;
    tensor->buff = CCML_BUFF_CNST;

    int size = ccml_size(tensor);
    tensor->data = ccml_malloc(ctx, size * sizeof(float));
    for (int i = 0; i < size; i++) {
        *((float *)tensor->data + i) = data[i];
    }
}

CCML_API void ccml_fill(ccml_context * ctx, ccml_tensor * tensor, float value) {
    tensor->type = CCML_TYPE_FP32;
    tensor->buff = CCML_BUFF_CNST;

    int size = ccml_size(tensor);
    tensor->data = ccml_malloc(ctx, size * sizeof(float));
    for (int i = 0; i < size; i++) {
        *((float *)tensor->data + i) = value;
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

    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        if (lhs->shape[i] != rhs->shape[i] && lhs->shape[i] != 1 && rhs->shape[i] != 1)
            return false;
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
    return tensor->buff != CCML_BUFF_NEW;
}

CCML_API bool ccml_is_reshape(ccml_tensor * tensor) {
    return tensor->oper == CCML_OPER_RES || tensor->oper == CCML_OPER_PER;
}

CCML_API int ccml_ndim(ccml_tensor * tensor) {
    int last_dim = 0;
    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        if (tensor->shape[i] != 1) last_dim = i;
    }

    return last_dim == 0 ? 1 : last_dim + 1;
}

CCML_API bool ccml_is_vector(ccml_tensor * tensor) {
    return tensor->shape[0] != 1 && tensor->shape[1] == 1 && tensor->shape[2] == 1 && tensor->shape[3] == 1;
}

CCML_API bool ccml_is_matrix(ccml_tensor * tensor) {
    return tensor->shape[0] != 1 && tensor->shape[1] != 1 && tensor->shape[2] == 1 && tensor->shape[3] == 1;
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

CCML_API ccml_tensor * ccml_log(ccml_context * ctx, ccml_tensor * tensor) {
    ccml_tensor * result = ccml_new_tensor_impl(
        ctx, tensor->type, tensor->shape, tensor, NULL, tensor->is_grad);
    result->oper = CCML_OPER_LOG;

    return result;
}

CCML_API ccml_tensor * ccml_exp(ccml_context * ctx, ccml_tensor * tensor) {
    ccml_tensor * result = ccml_new_tensor_impl(
        ctx, tensor->type, tensor->shape, tensor, NULL, tensor->is_grad);
    result->oper = CCML_OPER_EXP;

    return result;
}

CCML_API ccml_tensor * ccml_sin(ccml_context * ctx, ccml_tensor * tensor) {
    ccml_tensor * result = ccml_new_tensor_impl(
        ctx, tensor->type, tensor->shape, tensor, NULL, tensor->is_grad);
    result->oper = CCML_OPER_SIN;

    return result;
}

CCML_API ccml_tensor * ccml_rec(ccml_context * ctx, ccml_tensor * tensor) {
    ccml_tensor * result = ccml_new_tensor_impl(
        ctx, tensor->type, tensor->shape, tensor, NULL, tensor->is_grad);
    result->oper = CCML_OPER_REC;

    return result;
}

CCML_API ccml_tensor * ccml_sqrt(ccml_context * ctx, ccml_tensor * tensor) {
    ccml_tensor * result = ccml_new_tensor_impl(
        ctx, tensor->type, tensor->shape, tensor, NULL, tensor->is_grad);
    result->oper = CCML_OPER_SQRT;

    return result;
}

CCML_API ccml_tensor * ccml_add(ccml_context * ctx, ccml_tensor * lhs, ccml_tensor * rhs) {
    CCML_ASSERT(ccml_can_broadcast(lhs, rhs), "incompatible dimensions for broadcasting");

    bool null_input = lhs == NULL || rhs == NULL;

    int shape[CCML_DIMS_MAX] = {0};
    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        shape[i] = null_input ? lhs->shape[i] :
        (lhs->shape[i] + rhs->shape[i] + abs(lhs->shape[i] - rhs->shape[i])) / 2;
    }

    bool is_grad = lhs->is_grad || (null_input ? false : rhs->is_grad);
    ccml_tensor * result = ccml_new_tensor_impl(ctx, lhs->type, shape, lhs, rhs, is_grad);
    result->oper = CCML_OPER_ADD;

    return result;
}

CCML_API ccml_tensor * ccml_mul(ccml_context * ctx, ccml_tensor * lhs, ccml_tensor * rhs) {
    CCML_ASSERT(ccml_can_broadcast(lhs, rhs), "incompatible dimensions for broadcasting");

    bool null_input = lhs == NULL || rhs == NULL;

    int shape[CCML_DIMS_MAX] = {0};
    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        shape[i] = null_input ? lhs->shape[i] :
        (lhs->shape[i] + rhs->shape[i] + abs(lhs->shape[i] - rhs->shape[i])) / 2;
    }

    bool is_grad = lhs->is_grad || (null_input ? false : rhs->is_grad);
    ccml_tensor * result = ccml_new_tensor_impl(ctx, lhs->type, shape, lhs, rhs, is_grad);
    result->oper = CCML_OPER_MUL;

    return result;
}

CCML_API ccml_tensor * ccml_reshape(ccml_context * ctx, ccml_tensor * tensor, int * shape) {
    int size = ccml_size(tensor);
    int new_size = shape[0] * shape[1] * shape[2] * shape[3];

    CCML_ASSERT(size == new_size, "reshaped and source tensor must have the same size");

    int stride[CCML_DIMS_MAX] = {
        shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1
    };

    ccml_tensor * result = ccml_new_tensor_impl(
        ctx, tensor->type, shape, tensor, NULL, tensor->is_grad);

    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        result->shape[i] = shape[i];
        result->stride[i] = stride[i];
    }

    result->buff = CCML_BUFF_SAVE;
    result->oper = CCML_OPER_RES;
    result->data = tensor->data;

    return result;
}

CCML_API ccml_tensor * ccml_permute(ccml_context * ctx, ccml_tensor * tensor, int * perm) {
    ccml_tensor * result = ccml_new_tensor_impl(
        ctx, tensor->type, tensor->shape, tensor, NULL, tensor->is_grad);

    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        result->shape[i] = tensor->shape[perm[i]];
        result->stride[i] = tensor->stride[perm[i]];
    }

    result->buff = CCML_BUFF_SAVE;
    result->oper = CCML_OPER_PER;
    result->data = tensor->data;

    return result;
}

CCML_API ccml_tensor * ccml_sum(ccml_context * ctx, ccml_tensor * tensor, int n_axes, int * axes) {
    CCML_ASSERT(n_axes >= 0 && n_axes <= CCML_DIMS_MAX, "invalid number of summed axes");

    int shape[CCML_DIMS_MAX] = {1, 1, 1, 1};
    for (int i = 0; i < CCML_DIMS_MAX; i++) {
        shape[i] = tensor->shape[i];
    }
    for (int i = 0; i < n_axes; i++) {
        shape[axes[i]] = 1;
    }

    ccml_tensor * result = ccml_new_tensor_impl(
        ctx, tensor->type, shape, tensor, NULL, tensor->is_grad);

    result->oper = CCML_OPER_SUM;
    result->buff = CCML_BUFF_SAVE;

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

CCML_API ccml_tensor * ccml_neg(ccml_context * ctx, ccml_tensor * tensor) {
    return ccml_log(ctx, ccml_rec(ctx, ccml_exp(ctx, tensor)));
}

CCML_API ccml_tensor * ccml_sub(ccml_context * ctx, ccml_tensor * lhs, ccml_tensor * rhs) {
    return ccml_add(ctx, lhs, ccml_neg(ctx, rhs));
}

CCML_API ccml_tensor * ccml_div(ccml_context * ctx, ccml_tensor * lhs, ccml_tensor * rhs) {
    return ccml_mul(ctx, lhs, ccml_rec(ctx, rhs));
}

CCML_API ccml_tensor * ccml_square(ccml_context * ctx, ccml_tensor * tensor) {
    return ccml_mul(ctx, tensor, tensor);
}

CCML_API ccml_tensor * ccml_cos(ccml_context * ctx, ccml_tensor * tensor) {
    int shape[CCML_DIMS_MAX] = {1, 1, 1, 1};
    ccml_tensor * pi_2 = ccml_new_tensor_impl(
        ctx, tensor->type, shape, NULL, NULL, false);

    ccml_fill(ctx, pi_2, M_PI_2);
    return ccml_sin(ctx, ccml_add(ctx, tensor, pi_2));
}

CCML_API ccml_tensor * ccml_tanh(ccml_context * ctx, ccml_tensor * tensor) {
    return ccml_div(ctx, ccml_sub(ctx, ccml_exp(ctx, tensor), ccml_exp(ctx, ccml_neg(ctx, tensor))),
           ccml_add(ctx, ccml_exp(ctx, tensor), ccml_exp(ctx, ccml_neg(ctx, tensor))));
}

CCML_API ccml_tensor * ccml_matmul(ccml_context * ctx, ccml_tensor * lhs, ccml_tensor * rhs) {
    CCML_ASSERT(ccml_is_matrix(lhs));
    CCML_ASSERT(ccml_is_matrix(rhs));
    CCML_ASSERT(lhs->shape[1] == rhs->shape[0]);

    ccml_tensor * lhs_r = ccml_reshape(ctx, lhs, (int[]){lhs->shape[0], lhs->shape[1], 1, 1});
    ccml_tensor * rhs_r = ccml_reshape(ctx, rhs, (int[]){1, rhs->shape[0], rhs->shape[1], 1});

    ccml_tensor * mul = ccml_mul(ctx, lhs_r, rhs_r);
    ccml_tensor * sum = ccml_sum(ctx, mul, 1, (int[]){1});
    ccml_tensor * res = ccml_reshape(ctx, sum, (int[]){sum->shape[0], sum->shape[2], 1, 1});

    return res;
}

CCML_API ccml_tensor * ccml_soft_max(ccml_context * ctx, ccml_tensor * tensor) {
    int dims[CCML_DIMS_MAX] = {0, 1, 2, 3};
    return ccml_div(ctx, ccml_exp(ctx, tensor), ccml_sum(ctx, ccml_exp(ctx, tensor), 4, dims));
}

CCML_API ccml_tensor * ccml_cross_entropy_loss(ccml_context * ctx, ccml_tensor * tensor, ccml_tensor * target) {
    CCML_ASSERT(ccml_size(tensor) == ccml_size(target));

    int dims[CCML_DIMS_MAX] = {0, 1, 2, 3};
    return ccml_neg(ctx, ccml_sum(ctx, ccml_mul(ctx, target, ccml_log(ctx, tensor)), 4, dims));
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

CCML_API ccml_hashmap * ccml_new_hashmap(ccml_context * ctx) {
    int capacity = CCML_NODE_MAX;
    ccml_hashmap * map = ccml_malloc(ctx, sizeof(ccml_hashmap));

    *map = (ccml_hashmap) {
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
    if (key == NULL) return -1;

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
    if (map->used >= map->capacity) CCML_ASSERT(false, "hashmap size overflow");

    uint64_t hash = ccml_hash_key(key);
    int index = (int)(hash & (int)(map->capacity - 1));

    while (map->entries[index].key != 0) {
        if ((uintptr_t)key == map->entries[index].key) {
            // Found key (it already exists), update value.
            map->entries[index].value = value;
            return;
        }
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
    ccml_context * ctx;
} ccml_graph;

CCML_API void ccml_graph_forward(ccml_graph * graph, ccml_tensor * tensor, int * node_counter) {
    if (tensor == NULL) return;

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

CCML_API void ccml_graph_backward(ccml_context * ctx, ccml_graph * graph, ccml_tensor * root) {
    if (root->is_grad == false) return;
    int shape[CCML_DIMS_MAX] = {1, 1, 1, 1};
    root->grad = ccml_new_tensor_impl(ctx, root->type, shape, NULL, NULL, false);
    ccml_fill(ctx, root->grad, 1.0f);

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
        if (tensor->is_grad == true) {

            int n_sum_dims_0 = 0;
            int n_sum_dims_1 = 0;
            int sum_dims_0[CCML_DIMS_MAX] = {1, 1, 1, 1};
            int sum_dims_1[CCML_DIMS_MAX] = {1, 1, 1, 1};

            for (int i = 0; i < CCML_DIMS_MAX; i++) {
                if (tensor->src[0] != NULL && tensor->src[0]->shape[i] == 1 && tensor->shape[i] != 1) {
                    sum_dims_0[n_sum_dims_0++] = i;
                }
                if (tensor->src[1] != NULL && tensor->src[1]->shape[i] == 1 && tensor->shape[i] != 1) {
                    sum_dims_1[n_sum_dims_1++] = i;
                }
            }

            // calculating partials
            ccml_tensor * grad_0 = NULL;
            ccml_tensor * grad_1 = NULL;

            switch (tensor->oper) {
                case CCML_OPER_NEW:
                    grad_0 = tensor->grad; break;
                case CCML_OPER_COPY:
                    grad_0 = tensor->grad; break;
                case CCML_OPER_LOG:
                    grad_0 = ccml_mul(ctx, tensor->grad, ccml_rec(ctx, tensor->src[0])); break;
                case CCML_OPER_EXP:
                    grad_0 = ccml_mul(ctx, tensor->grad, ccml_exp(ctx, tensor->src[0])); break;
                case CCML_OPER_SIN:
                    grad_0 = ccml_mul(ctx, tensor->grad, ccml_cos(ctx, tensor->src[0])); break;
                case CCML_OPER_REC:
                    grad_0 = ccml_mul(ctx, tensor->grad, ccml_neg(ctx, ccml_rec(ctx, ccml_square(ctx, tensor->src[0])))); break;
                case CCML_OPER_SQRT: {
                    int shape[CCML_DIMS_MAX] = {1, 1, 1, 1};
                    ccml_tensor * two = ccml_new_tensor_impl(ctx, tensor->type, shape, NULL, NULL, false);
                    ccml_fill(ctx, two, 2.0f);
                    grad_0 = ccml_mul(ctx, tensor->grad, ccml_rec(ctx, ccml_mul(ctx, two, ccml_sqrt(ctx, tensor->src[0])))); break; }
                case CCML_OPER_ADD:
                    grad_0 = ccml_sum(ctx, tensor->grad, n_sum_dims_0, sum_dims_0);
                    grad_1 = ccml_sum(ctx, tensor->grad, n_sum_dims_1, sum_dims_1); break;
                case CCML_OPER_MUL:
                    grad_0 = ccml_sum(ctx, ccml_mul(ctx, tensor->grad, tensor->src[1]), n_sum_dims_0, sum_dims_0);
                    grad_1 = ccml_sum(ctx, ccml_mul(ctx, tensor->grad, tensor->src[0]), n_sum_dims_1, sum_dims_1); break;
                case CCML_OPER_SUM:
                case CCML_OPER_PER:
                case CCML_OPER_RES: {
                    int * shape = tensor->src[0]->shape;
                    ccml_tensor * one = ccml_new_tensor_impl(ctx, tensor->type, shape, NULL, NULL, false);
                    ccml_fill(ctx, one, 1.0f);
                    grad_0 = ccml_mul(ctx, tensor->grad, one); break; }
                default: CCML_ASSERT(false, "unknown variant of ccml_oper");
            }

            // multiplying tensor->grad by partials and adding them to
            // the gradients of the tensor's children (we have to do a
            // mini DFS traversal w/ ccml_graph_forward() since the gradient
            // calculation forms a mini sub-graph that needs to be tra-
            // versed separately)

            if (tensor->src[0] != NULL && tensor->src[0]->is_grad) {
                if (ccml_is_leaf(tensor->src[0])) {
                    tensor->src[0]->grad = ccml_add(ctx, grad_0, tensor->src[0]->grad);
                    ccml_fill(ctx, tensor->src[0]->grad, 0.0f);
                } else {
                    tensor->src[0]->grad = ccml_mul(ctx, grad_0, tensor->src[0]->grad);
                    ccml_fill(ctx, tensor->src[0]->grad, 1.0f);
                }
                ccml_graph_forward(graph, tensor->src[0]->grad, &graph->n_nodes);
                queue[queue_end++] = tensor->src[0];
            }

            if (tensor->src[1] != NULL && tensor->src[1]->is_grad) {
                if (ccml_is_leaf(tensor->src[1])) {
                    tensor->src[1]->grad = ccml_add(ctx, grad_1, tensor->src[1]->grad);
                    ccml_fill(ctx, tensor->src[1]->grad, 0.0f);
                } else {
                    tensor->src[1]->grad = ccml_mul(ctx, grad_1, tensor->src[1]->grad);
                    ccml_fill(ctx, tensor->src[1]->grad, 1.0f);
                }
                ccml_graph_forward(graph, tensor->src[1]->grad, &graph->n_nodes);
                queue[queue_end++] = tensor->src[1];
            }
        }
    }
}

CCML_API void ccml_graph_allocate(ccml_context * ctx, ccml_graph * graph) {
    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        int size = ccml_size(tensor);
        if (ccml_has_buffer(tensor) && tensor->data == NULL) {
            tensor->data = ccml_malloc(ctx, size * ccml_type_sizes[tensor->type]);
        }
        if (tensor->grad != NULL && ccml_has_buffer(tensor)) {
            tensor->grad->buff = CCML_BUFF_SAVE;
        }
    }
}

//
//   ██████╗ ██████╗ ████████╗██╗███╗   ███╗██╗███████╗███████╗
//  ██╔═══██╗██╔══██╗╚══██╔══╝██║████╗ ████║██║██╔════╝██╔════╝
//  ██║   ██║██████╔╝   ██║   ██║██╔████╔██║██║███████╗█████╗
//  ██║   ██║██╔═══╝    ██║   ██║██║╚██╔╝██║██║╚════██║██╔══╝
//  ╚██████╔╝██║        ██║   ██║██║ ╚═╝ ██║██║███████║███████╗
//   ╚═════╝ ╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝╚══════╝╚══════╝
//

CCML_API void ccml_graph_cse(ccml_graph * graph) {
    for (int i = 0; i < graph->map->capacity; i++) {
        graph->map->used = 0;
        graph->map->entries[i].key = 0;
        graph->map->entries[i].value = -1;
    }

    // an expression takes the following format of an integer
    // (oper->int) * 10^6 + (src0->index) * 10^3 + (src1->index)
    // with absent/NULL children being assigned index CCML_NODE_MAX

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
                tensor->oper = CCML_OPER_COPY;
                tensor->src[0] = graph->nodes[index];
                tensor->src[1] = NULL;
            }
        }
    }
}

CCML_API ccml_graph * ccml_new_graph(ccml_context * ctx, ccml_tensor * root) {
    struct ccml_graph * graph = ccml_malloc(ctx, sizeof(struct ccml_graph));
    root->buff = CCML_BUFF_SAVE;

    *graph = (struct ccml_graph) {
        /*.n_nodes =*/ 0,
        /*.nodes   =*/ {NULL},
        /*.map     =*/ ccml_new_hashmap(ctx),
        /*.context =*/ ctx
    };

    ccml_graph_forward(graph, root, &graph->n_nodes);
    ccml_graph_backward(ctx, graph, root);
    ccml_graph_cse(graph);
    ccml_graph_allocate(ctx, graph);

    return graph;
};

//
//  ██╗███╗   ██╗██████╗ ███████╗██╗  ██╗
//  ██║████╗  ██║██╔══██╗██╔════╝╚██╗██╔╝
//  ██║██╔██╗ ██║██║  ██║█████╗   ╚███╔╝
//  ██║██║╚██╗██║██║  ██║██╔══╝   ██╔██╗
//  ██║██║ ╚████║██████╔╝███████╗██╔╝ ██╗
//  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝
//

CCML_API const char * ccml_new_index(ccml_context * ctx, ccml_tensor * parent, ccml_tensor * child) {
    int size = CCML_CHAR_MAX;
    char * index = ccml_malloc(ctx, size * sizeof(char));
    *index = '\0';
    strncat(index, "[", size);

    for (int i = 0; i < ccml_ndim(child); i++) {
        bool dim_is_fake = false;
        if (parent != NULL) {
            dim_is_fake = parent->shape[i] != 1 && child->shape[i] == 1;
        }

        snprintf(index + strlen(index), size - strlen(index), "%sid%d*%d*%d",
                 i != 0 && i != ccml_ndim(child) ? "+" : "", i, child->stride[i],
                 dim_is_fake ? 0 : 1);
    }

    strncat(index + strlen(index), "]", size - strlen(index));
    // if tensor isn't indexiable, it's a scalar, we overwrite the index
    // string with nothing (empty)
    if (ccml_is_indexable(child) == false) {
        snprintf(index, size, "");
    }

    return index;
}

CCML_API void ccml_kernel_slicing(
    ccml_graph * graph, int * n_kernels, int kernels[CCML_KERN_MAX][2]) {
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
        case CCML_OPER_LOG: return "log";
        case CCML_OPER_EXP: return "exp";
        case CCML_OPER_SIN: return "sin";
        case CCML_OPER_REC: return "1/";
        case CCML_OPER_SQRT: return "sqrt";
        case CCML_OPER_ADD: return "+";
        case CCML_OPER_MUL: return "*";
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
    switch (type) {
        case CCML_TYPE_FP32: return "Float";
        default: CCML_ASSERT(false, "unknown variant of ccml_type");
    }
}

CCML_API const char * ccml_kernel_metal(
    ccml_context * ctx, struct ccml_graph * graph, int n_kernel, int start, int finish) {
    int size = CCML_NODE_MAX * CCML_CHAR_MAX;
    char * kernel = ccml_malloc(ctx, size * sizeof(char));
    char * string = kernel;
    *string = '\0';

    // the n_kernel parameter specifies the index of the kernel being generated,
    // and start to finish are indexes of graph->nodes that pertain to that kernel

    if (n_kernel == 0) {
        string += snprintf(string, size - (kernel - string), "#include <metal_stdlib>\nusing namespace metal;\n");
    }

    string += snprintf(string, size - (kernel - string), "kernel void my_kernel_%d(", n_kernel);

    // adding kernel input parameters to the kernel string
    int n_kernel_parameters = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];

        if (ccml_is_indexable(tensor) && ccml_is_reshape(tensor) == false) {
            if (n_kernel_parameters == 0) {
                string += snprintf(string, size - (kernel - string), "device %s* data_%d [[buffer(0)]]",
                                   ccml_type_metal(tensor->type), i);
                n_kernel_parameters++;
            } else if (ccml_is_reshape(tensor) == false) {
                string += snprintf(string, size - (kernel - string), ", device %s* data_%d [[buffer(%d)]]",
                                   ccml_type_metal(tensor->type), i, n_kernel_parameters);
                n_kernel_parameters++;
            }
        }
    }

    int grid[CCML_DIMS_MAX] = {1, 1, 1, 1};
    for (int i = start; i < finish; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        grid[0] = grid[0] > tensor->shape[0] ? grid[0] : tensor->shape[0];
        grid[1] = grid[1] > tensor->shape[1] ? grid[1] : tensor->shape[1];
    }

    string += snprintf(string, size - (kernel - string),
                       ", uint3 gid [[thread_position_in_grid]]) {\n"
                       "\tuint id0 = gid.x / %d;\n\tuint id1 = gid.x %% %d;\n"
                       "\tuint id2 = gid.y;\n\tuint id3 = gid.z;\n\n",
                       grid[1], grid[1]);

    for (int i = start; i < finish; i++) {
        ccml_tensor * tensor = graph->nodes[i];

        switch (tensor->oper) {
        case CCML_OPER_NEW:
            // tensor data is embeddeable directly into the kernel string
            if (ccml_has_buffer(tensor) && ccml_is_indexable(tensor) == false) {
                string += snprintf(string, size - (kernel - string), "\t%sdata_%d = %f;\n",
                                   ccml_type_metal(tensor->type), i, *(float *)tensor->data);
            }
            break;
        case CCML_OPER_COPY:
            string += snprintf(string, size - (kernel - string),
                               "\t%s%s data_%d%s = data_%d%s;\n", ccml_type_metal(tensor->type),
                               ccml_is_indexable(tensor) ? "*" : "", tensor->index,
                               ccml_new_index(ctx, NULL, tensor), tensor->src[0]->index,
                               ccml_new_index(ctx, tensor, tensor->src[0]));
            break;
        case CCML_OPER_LOG:
        case CCML_OPER_EXP:
        case CCML_OPER_SIN:
        case CCML_OPER_REC:
        case CCML_OPER_SQRT:
            string += snprintf(string, size - (kernel - string), "\t%sdata_%d%s = ",
                               ccml_is_indexable(tensor) ? "" : ccml_type_metal(tensor->type),
                               tensor->index, ccml_new_index(ctx, NULL, tensor));
            string += snprintf(string, size - (kernel - string), "%s(data_%d%s);\n",
                               ccml_oper_metal(tensor->oper), tensor->src[0]->index,
                               ccml_new_index(ctx, tensor, tensor->src[0]));
            break;
        case CCML_OPER_ADD:
        case CCML_OPER_MUL:
            string += snprintf(string, size - (kernel - string), "\t%sdata_%d%s = ",
                               ccml_is_indexable(tensor) ? "" : ccml_type_metal(tensor->type),
                               tensor->index, ccml_new_index(ctx, NULL, tensor));
            string += snprintf(string, size - (kernel - string), "data_%d%s %s ",
                               tensor->src[0]->index, ccml_new_index(ctx, tensor, tensor->src[0]),
                               ccml_oper_metal(tensor->oper));
            string += snprintf(string, size - (kernel - string), "data_%d%s;\n",
                               tensor->src[1] != NULL ? tensor->src[1]->index : tensor->index,
                               tensor->src[1] != NULL ? ccml_new_index(ctx, tensor, tensor->src[1])
                               : ccml_new_index(ctx, NULL, tensor));
            break;
        case CCML_OPER_SUM:
            string += snprintf(string, size - (kernel - string), "\tfor (int i = 0; i < %d; i++) {\n\t\tdata_%d%s += ",
                               ccml_size(tensor->src[0]) / ccml_size(tensor), tensor->index,
                               ccml_new_index(ctx, NULL, tensor));
            string += snprintf(string, size - (kernel - string), "data_%d%s;\n\t}\n",
                               tensor->src[0]->index, ccml_new_index(ctx, tensor, tensor->src[0]));
            break;
        case CCML_OPER_RES:
        case CCML_OPER_PER:
            string += snprintf(string, size - (kernel - string), "\tdevice %s* data_%d = data_%d;\n",
                               ccml_type_metal(tensor->type), tensor->index, tensor->src[0]->index);
            break;
        default:
            CCML_ASSERT(false, "unknown variant of ccml_oper");
        }
    }

    string += snprintf(string, size - (kernel - string), "}");
    return kernel;
}

CCML_API void ccml_setup_metal(ccml_graph * graph) {
    FILE * file_ptr = fopen("metal.swift", "w");
    CCML_ASSERT(file_ptr != NULL, "failed to create file for metal source");

    FILE * file_kernel_ptr = fopen("kernel.metal", "w");
    CCML_ASSERT(file_kernel_ptr != NULL, "failed to create a file for kernel source");

    // kernels[i][0] and kernels[i][1] specifies the starting and finishing
    // indexes of graph->nodes that pertain to i-th kernel
    int kernels[CCML_KERN_MAX][2] = {0};
    int n_kernels = 0;
    ccml_kernel_slicing(graph, &n_kernels, kernels);

    // general metal setup
    fprintf(file_ptr, "import MetalKit\n\n"
            "let device = MTLCreateSystemDefaultDevice()!\n"
            "let filePath = Bundle.main.path(forResource: \"kernel\", ofType: "
            "\"metal\")!\n"
            "let source = try! String(contentsOfFile: filePath, encoding: .utf8)\n"
            "let library = try! device.makeLibrary(source: source, options: nil)\n"
            "let commandQueue = device.makeCommandQueue()!\n\n");

    for (int i = 0; i < graph->n_nodes; i++) {
        ccml_tensor * tensor = graph->nodes[i];
        if (ccml_is_indexable(tensor) && ccml_is_reshape(tensor) == false) {
            fprintf(file_ptr, "var cpu_buffer_%d: [%s] = [%f", i,
                    ccml_type_swift(tensor->type), *(float *)tensor->data);
            for (int j = 1; j < ccml_size(tensor); j++) {
                fprintf(file_ptr, ", %f", *((float *)tensor->data + j));
            }
            fprintf(file_ptr, "]\n");
            fprintf(file_ptr, "let gpu_buffer_%d = device.makeBuffer(bytes: cpu_buffer_%d, "
                    "length: %d * MemoryLayout<%s>.size, options: [])\n",
                    i, i, ccml_size(tensor), ccml_type_swift(tensor->type));
        }
    }

    // code specific to each i-th kernel
    for (int i = 0; i < n_kernels; i++) {
        fprintf(file_kernel_ptr, "%s\n\n", ccml_kernel_metal(graph->ctx, graph, i, kernels[i][0], kernels[i][1]));
        fprintf(file_ptr, "\n// setting up my_kernel_%d\n"
                "let kernelFunction_%d  = library.makeFunction(name: "
                "\"my_kernel_%d\")!\n"
                "let computePipeline_%d = try! "
                "device.makeComputePipelineState(function: kernelFunction_%d)\n"
                "let commandBuffer_%d   = commandQueue.makeCommandBuffer()!\n"
                "let computeEncoder_%d  = "
                "commandBuffer_%d.makeComputeCommandEncoder()!\n"
                "computeEncoder_%d.setComputePipelineState(computePipeline_%d)\n\n",
                i, i, i, i, i, i, i, i, i, i);

        // code specific to i-th kernel's buffers
        int thread_grid[CCML_DIMS_MAX] = {1, 1, 1, 1};
        int n_buffers = 0;
        for (int j = 0; j < graph->n_nodes; j++) {
            ccml_tensor * tensor = graph->nodes[j];
            if (ccml_is_indexable(tensor) && ccml_is_reshape(tensor) == false) {
                // selecting the largest dimensions for thread grid out of nodes
                // that belong to i-th kernel
                if (j >= kernels[i][0] && j <= kernels[i][1]) {
                    thread_grid[0] = thread_grid[0] > tensor->shape[0] ? thread_grid[0] : tensor->shape[0];
                    thread_grid[1] = thread_grid[1] > tensor->shape[1] ? thread_grid[1] : tensor->shape[1];
                    thread_grid[2] = thread_grid[2] > tensor->shape[2] ? thread_grid[2] : tensor->shape[2];
                    thread_grid[3] = thread_grid[3] > tensor->shape[3] ? thread_grid[3] : tensor->shape[3];
                }

                fprintf(file_ptr, "computeEncoder_%d.setBuffer(gpu_buffer_%d, offset: 0, "
                        "index: %d)\n", i, j, n_buffers++);
            }
        }

        fprintf(file_ptr, "\nlet gridSize_%d = MTLSize(width: %d, height: %d, depth: %d)\n"
                "let threadGroupSize_%d = MTLSize(width: 1, height: 1, depth: 1)\n"
                "computeEncoder_%d.dispatchThreads(gridSize_%d, "
                "threadsPerThreadgroup: threadGroupSize_%d)\n"
                "computeEncoder_%d.endEncoding()\n"
                "commandBuffer_%d.commit()\n"
                "commandBuffer_%d.waitUntilCompleted()\n\n",
                i, thread_grid[0] * thread_grid[1], thread_grid[2], thread_grid[3], i, i, i, i, i, i, i);

        // copying data back from GPU to buffers (specific to i-th kernel)
        for (int j = kernels[i][0]; j < kernels[i][1]; j++) {
            ccml_tensor * tensor = graph->nodes[j];
            if (ccml_is_indexable(tensor) && ccml_is_reshape(tensor) == false) {
                fprintf(file_ptr, "memcpy(&cpu_buffer_%d, gpu_buffer_%d!.contents(), "
                        "cpu_buffer_%d.count * MemoryLayout<%s>.size);\n"
                        "print(cpu_buffer_%d)\n", j, j, j, ccml_type_swift(tensor->type), j);
            }
        }
    }

    fclose(file_kernel_ptr);
    fclose(file_ptr);
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
    ccml_setup_metal(graph);

    #if defined(__APPLE__)
        system("swiftc -o metal metal.swift -framework MetalKit && ./metal");
        system("rm metal");
    #endif

    // system("rm kernel.metal metal.swift");
}

#endif /* CCML_IMPL */