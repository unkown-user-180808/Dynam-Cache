# Dynam-Cache

Official implementation of **Dynam-Cache: Geometry-Aware Temporal KV Reuse for Efficient Multi-View VLA Inference**.

Dynam-Cache is a training-free inference method that reduces VLA inference cost by selectively reusing visual KV states across consecutive observations. It is designed for multi-view manipulation with both fixed and moving wrist cameras.

This repository is built on [OpenVLA-OFT](https://github.com/moojink/openvla-oft) and evaluates Dynam-Cache on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).

## Overview

Dynam-Cache exploits temporal redundancy in multi-view robot observations while accounting for camera motion and task relevance.

The method combines:

- geometry-aware correspondence for the moving wrist camera,
- instruction-guided selection of reusable visual tokens,
- progressive KV reuse across transformer layers, and
- kinematics-guided reuse budgets for free motion and fine-grained manipulation.

Dynam-Cache is applied only at inference time and does not require retraining the VLA policy.