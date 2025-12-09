# PySIRTEM: An Efficient Modular Simulation Platform for the Analysis of Pandemic Scenarios

<p align="center">
  <img src="https://github.com/Preetom1905011/PySIRTEM/blob/main/code/PySIRTEM_overview1.png" alt="PySIRTEM Diagram" width="500"/>
</p>

## Paper
The full paper is available in this repository as **Paper.pdf**.  
All scripts used for experiments, simulations, and analysis are located in the **code/** folder.

## Abstract
Conventional population-based ODE models struggle as model resolution increases, since incorporating many states exponentially raises computational costs and demands robust calibration of numerous hyperparameters. PySIRTEM is a spatiotemporal SEIR-based epidemic simulation platform that provides high-resolution analysis of viral disease progression and mitigation. Based on the authors’ MATLAB simulator SIRTEM, PySIRTEM’s modular design reflects key health processes—including infection, testing, immunity, and hospitalization—enabling flexible manipulation of transition rates. Unlike SIRTEM, PySIRTEM uses a Sequential Monte Carlo (SMC) particle filter to dynamically learn epidemiological parameters using historical COVID-19 data from several U.S. states. The improved accuracy (by orders of magnitude) makes PySIRTEM well-suited for
