<div align="center">
  
# ddsp_textures_thesis

[**Esteban Gutiérrez**](https://github.com/cordutie)<sup>1</sup> and [**Lonce Wyse**](https://lonce.org/)<sup>1</sup>

<sup>1</sup> *Department of Information and Communications Technologies, Universitat Pompeu Fabra* <br>

<div align="left">

## 1. Introduction

This repository contains the experiments conducted for the thesis titled "Statistics-Driven Texture Sound Synthesis Using Differentiable Digital Signal Processing-Based Architectures" authored by Esteban Gutiérrez and advised by Lonce Wyse at the [Universitat Pompeu Fabra](https://upf.edu). The experiments are based on the GitHub repositories resulting from this thesis: [ddsp_textures](https://github.com/cordutie/ddsp_textures) and [VAE_SubEnv](https://github.com/cordutie/VAE_SubEnv).

![DDSP architecture](experiments/data/images/DDSP.png)
<div align="center">Figure 1. DDSP architecture modified to synthesize texture sounds.</div><br>

The thesis explores adapting Differentiable Digital Signal Processing (DDSP) architectures, first introduced by Engel et al. in [[1]](#1), for synthesizing and controlling texture sounds, which are complex and noisy compared to traditional pitched instrument timbres. It introduces two innovative synthesizers: the $\texttt{SubEnv\ Synth}$, which applies amplitude envelopes to subband decompositions of white noise, and the $\texttt{P-VAE\ Synth}$, which integrates a Poisson process with a Variational Autoencoder (VAE) to handle time and event-based aspects of texture sounds based on the early conceptions of a texture sound introduced by Saint-Arnaud in [[2]](#2). Additionally, the $\texttt{TextStat}$ loss function is presented, inspired in McDermott and Simoncelli's work [[3]](#3) and designed to evaluate texture sounds based on their statistical properties rather than short-term perceptual similarity. The thesis demonstrates the application of these synthesizers and the loss function within DDSP-based frameworks, highlights mixed success in resynthesizing texture sounds, and identifies challenges, particularly with the $\texttt{P-VAE\ Synth}$. Future work will focus on optimizing the $\texttt{TextStat}$ loss function, reassessing the VAE component, and exploring real-time implementations. This research lays the groundwork for advancing texture sound synthesis and provides valuable insights for both theoretical and practical developments in audio signal processing.

![Latent space exploration](experiments/data/images/P-VAE.png)
<div align="center">Figure 2. Latent space exploration.</div><br>

## 2. Experiments

The `experiments` folder contains several Jupyter notebooks detailing the experiments conducted for this thesis. Below is a description of each subfolder:

### 2.1. Signal Processors

This section introduces two differentiable signal processors explored in this thesis. Details and experiments related to these processors can be found in the corresponding Jupyter notebooks.

### 2.2. Loss Functions

In this section, a new loss function is introduced and preliminary research is conducted to assess its effectiveness. The relevant notebooks provide insights into the performance and efficiency of this loss function.

### 2.3. DDSP

This subfolder contains Jupyter notebooks on models based on Differentiable Digital Signal Processing (DDSP) using the previously defined signal processors and loss function. These notebooks cover the training, exploration, and evaluation of these DDSP-based models.

### 2.4. New Models

Here, you’ll find notebooks detailing the training and evaluation of new model variants derived from the ones introduced in the thesis. These models are examined for their performance and effectiveness in the experiments.

## 3. References

<a id="1">[1]</a> J. Engel, L. Hantrakul, C. Gu, and A. Roberts, “Ddsp: Differentiable digital signal processing,” in International Conference on Learning Representations, 2020.\
<a id="2">[2]</a> N. Saint-Arnaud, “Classification of Sound Textures,” Master’s thesis, Massachusetts Institute of Technology, Cambridge, MA, 1995.\
<a id="3">[3]</a> J. H. McDermott and E. P. Simoncelli, “Sound texture perception via statistics of the auditory periphery: evidence from sound synthesis,” Neuron, vol. 71, pp. 926–940, 2011.\
