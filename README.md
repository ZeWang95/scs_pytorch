# scs_pytorch

Pytorch implementation of the Sharpened Cosine Similarity layer mentioned in a [tweet](https://twitter.com/_brohrer_/status/1487928061240946688).

The current version supports square kernels with arbitrary kernel sizes. 

Part of the implementation borrows from [dvisockas/cos_sim](https://github.com/dvisockas/cos_sim).

### To do

- [ ] depth-wise conv (with groups arguement)
- [ ] non-square kernels
- [ ] efficiency
- [ ] demo training code