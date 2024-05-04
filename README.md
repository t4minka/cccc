<p align="center">
    <img src="assets/logo.png" style="width: 60%" /><br/><br/>Single header
    automatic differentiation library in C.<br/><br/>
</p>

### What?

✳️ CCML is inspired by libraries like tinygrad and luminal that define complex operations in terms of a set of primitive operations, enabling aggressive operation fusion, automatic compute kernel generation, and more. Check out examples folder for usage scenarios.

### How?

✳️ CCML takes user defined operations like matrix multiplication or tangent, constructs a single computational graph with both forward and backward pass, applies some primitive optimisations and generates a single compute kernel to execute the graph.

### Why?

✳️ CCML is an educational project to help learn the implementation details of various concepts in compute science. The project is around 1k lines of code, making it easy to poke around and experiment with the it, while keeping it all in your head. 