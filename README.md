# C3D with Batch Normalization
Modified C3D[1] model introduced by Facebook AI Research by adding batch normalization[2] for regularization.
- The dropout in the original C3D architecture has removed.
- Input frame count = 16
- Input frame size = 112 x 112
- Input tensor shape = (Batch_Size, 3, 16, 112, 112)

References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.

[2] Ioffe, Surgey, et al. "Batch Normalization: Accelerating deep network training by reducing internal covariate shift."
arXiv:1502.03167v2 [cs.LG] 13 Feb 2015
