---
layout: page
title: TopoFuse
description: Topology-aware tri-planar fusion for 3D cryo-electron tomography segmentation. ECCV 2026.
importance: 1
category: research
---

**Carnegie Mellon University** · Advisors: Dr. Min Xu, Dr. Xingjian Li · May 2025 – Feb 2026

Automated segmentation of cryo-electron tomograms often produces masks that are voxel-accurate but topologically broken: membranes fragment, organelles merge and enclosed cavities collapse.

TopoFuse is a 3D volumetric segmentation framework that maps 2D features from a fine-tuned **SAM ViT-B** encoder into a 3D grid through **topology-conditioned Feature-wise Linear Modulation (FiLM) fusion**, weighing the reliability of each orthogonal view against missing-wedge artifacts. Topology is enforced through a differentiable, persistent-homology-guided projection rather than a soft loss penalty.

Across three cryo-ET benchmarks, TopoFuse reduces Betti number error by **54%** over the strongest soft-loss baseline and improves Dice by **4.6 points**, while editing only **3.1%** of voxels.

[Paper (arXiv)](https://arxiv.org/abs/2609.29717) · Accepted at the European Conference on Computer Vision (ECCV) 2026.
