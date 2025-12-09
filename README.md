# DAESIM

## Description

This repository contains the python implementation of the Dynamic Agro-Ecosystem SIMulator (DAESIM) model. 

Models for crop growth, development and yield are important tools for identifying opportunities for improved 
crop production and reduced environmental impacts in a changing world. This model provides a mechanistic 
description of physiological, structural, and developmental processes in annual crops. It includes an 
innovative, eco-evolutionary optimality approach to biomass partitioning (carbon allocation) that provides 
a foundation for evaluating crop performance in different climates and soil types. Key features of the plant 
model (DAESIM2-Plant) include: 
 * Mechanistic simulations of crop physiology and phenology.
 * Two-stream canopy radiative transfer.
 * Soil-plant-atmosphere hydraulic constraints.
 * Flexible temporal resolution (daily or custom step).
 * Optimal carbon allocation based on eco-evolutionary principles.

The model is written in a modular class-based Python architecture, with Jupyter notebook examples to 
test and evaluate modules either individually or in combination. 

Authors/Contributors: 
* Alexander Norton (CSIRO)
* Justin Borevitz (ANU)
* Firouzeh Taghikhah (USYD)

References: Taghikhah et al. (2022) https://doi.org/10.1016/j.ecolmodel.2022.109930

## Project Status and Scope

Currently, the model is used as a research tool as it is under active development. 

## Installation

Download the DAESim git repository onto your local machine by clicking on the green "Code" button and following your download method of choice. Once you have the DAESim repository on your local machine, you must install it by following the instructions below.

Before installation of DAESim, you will need to have the following things installed:
- pip: A Python package management tool. It is the recommended tool for installing and managing Python packages. 
- Anaconda (Conda): A cross-platform, package and environment management system. 
Conda and pip do a lot of similar things. Here, we use Conda to install Python package dependencies as well as to manage the virtual environment, while we use pip to install the DAESim package source code. 

## Getting started

Make your conda environment called `daesim` using the environment.yml file and the command:

$ conda env create --name daesim --file environment.yml

Then, you can activate your new environment with the command:

$ conda activate daesim

Now you have created the conda virtual environment to work in, we must install the package. To install the package use the command:

$ pip install -e .

Hopefully everything installed successfully. 

You can now run the code or open a jupyter notebook to start testing. To open a jupyter notebook run `jupyter-notebook` to open a Jupyter Notebook server or run `jupyter-lab` to open a JupyterLab server (recommended). From there, go into the notebooks directory and work through the examples.

Once you're finished, you can deactivate the conda environment with the command:

$ conda deactivate

### Why do we use Anaconda?

There are benefits to using Anaconda rather than just using pip. As discussed in this article (https://pythonspeed.com/articles/conda-vs-pip/) the main benefits are portability and reproducibility.
- Portability across operating systems: Instead of installing Python in three different ways on Linux, macOS, and Windows, you can use the same environment.yml on all three.
- Reproducibility: It’s possible to pin almost the whole stack, from the Python interpreter upwards.
- Consistent configuration: You don’t need to install system packages and Python packages in two different ways; (almost) everything can go in one file, the environment.yml.

Using Conda also addresses another problem: How to deal with Python libraries that require compiled code? At some point we may like to convert some of the DAESim source code into another, more computationally efficient language (e.g. C++ or Fortran) so that we can run larger simulations. If we decide to do that we will need to compile the source code in a language other than Python, in which case pip would be insufficient and Conda will be required. 

### Why do I only see `.py` files in the `notebooks` directory?

Tracking notebooks is a pain because of all the extra metadata that is in them.
As a result, we use [jupytext](https://jupytext.readthedocs.io/).
This allows us to track clean `.py` files while keeping all the power of jupyter notebooks.
You'll notice that we ignore `*.ipynb` in our `.gitignore`.

Thanks to jupytext, you can just click on the `.py` files and they will open like standard notebooks.

## Quick Start Guide

For a quick start it is recommended that you follow the instructions above to install the package and then step through 
the Jupyter notebook examples. 

## Configuration and Customisation

### Module-based Python architecture

The model is organised as a set of Python modules, each implemented as a class. Every module is designed to encapsulate:

1. Model parameters – as class attributes
2. Model functions – as class methods
3. (Optional) calculate method – which evaluates the rate of change in the state variables (the right-hand side of the ODEs) and diagnostic fluxes for that module

### Why this design?

This architecture is intentional and serves two main purposes:

1. Ease of use for model users
  * All parameters and methods for a process live in one place, so you can easily inspect, understand, and modify them from an interactive Python session, a script, or a notebook.
  * Parameters can be updated in-place (e.g. model.CanopyPhotosynthesis.g1 = 3.5) without needing to manage external parameter files or complex configuration layers.
  * This reduces friction for both exploratory experimentation and systematic calibration/sensitivity analysis.

2. Clear structure for model development
  * Each “model” component is effectively the combination of its parameters and methods, providing a clean separation between:
    * Model parameters (fixed traits and coefficients)
    * Model state variables (dynamic quantities integrated over time)
    * Model diagnostics (e.g. fluxes, intermediate rates)
    * Model forcing (external inputs: climate, soil, management)
  * This separation makes it easier to:
    * Swap or extend process formulations (e.g. new allocation scheme, alternative phenology model).
    * Trace how a change in one parameter or process propagates through the system.
    * Write tests for individual modules in isolation.

### Low-level and high-level modules

Modules are “stitched together” to represent the interacting components of the plant–soil–atmosphere system. Conceptually, we distinguish between:

* Low-level modules
  * Have few or no dependencies on other modules.
  * Implement fundamental relations and biophysics (e.g. solar geometry, canopy radiation, leaf gas exchange, soil hydraulic functions).
  * Can often be tested independently with minimal inputs.

* High-level modules
  * Depend on multiple low-level modules and coordinate interactions between them.
  * Typically implement emergent processes such as:
    * Whole-plant carbon and nitrogen balance
    * Optimal carbon allocation / biomass partitioning
    * Phenology and development
    * Yield formation
  * May include feedbacks, where:
    * Outputs from a high-level module (e.g. allocation decisions, canopy structure) are used as inputs to low-level modules (e.g. photosynthesis, transpiration, soil water uptake) in subsequent time steps.

This layered design makes it transparent how physiological, structural, and developmental processes are connected:

* You can analyse or replace a single process (e.g. the allocation scheme) without restructuring the entire codebase.
* You can run reduced models by using only a subset of modules.
* You can trace feedback loops (e.g. soil water → plant hydraulics → stomatal conductance → carbon gain → allocation → leaf/root growth → future water and carbon fluxes) in a structured and debuggable way.

### Customising parameters and processes

Because parameters live as class attributes, you can configure the model in several ways:

* At instantiation
* In-place, after creation.
* Via configuration helpers (if provided)


## General guidance on working in this repository

### Repository guiding principles

Text todo: Outline how to use this repository. e.g. /data/ should only contain a test set of forcing data, do not commit large datasets to it. Similarly, do not commit anything to /results/. Keep the repo clean and tidy. 

### Branch etiquette

In general, don't push to other people's branches (things can get weird if two people work on the same branch at once).
Instead, check out the branch and then create your own branch in case you want to make any commits (see "Checking out a branch related to a merge request" below).
Then, to add your changes, make a merge request from your own branch back into the original branch.

## Support

If you need support, the first place to go is the issue tracker (https://github.com/NortonAlex/DAESIM/issues).
From there, you can tag other model users to ask for help.
As a second step, reach out directly to the creators of the model.

## Other helpful snippets

### Checking out a branch related to a merge request

```sh
# Fetch any changes on the remote server
git fetch origin
# If you get no output, you're up to date
# If you get output, it will show you which branches have changed
# on the remote server

# Checkout the others' branch and create a local branch to track it
# git checkout -b branch-name origin/branch-name
# for example:
git checkout -b local-source origin/local-source

# Activate your environment
conda activate daesim

[remove]# Call make (just in case, you don't always have to do this)
# make conda-environment

# If any dependencies have changed, update your conda environment
conda env update --file environment.yml --prune

# Checkout (create) your own branch in case you want to make any commits
# (typically we use the branch name plus our initials)
git checkout -b test-notebook-an

# You're now ready to work
# E.g. by starting a notebook server and looking at the notebooks
jupyter notebook
```

## Guidance for good practice

A helpful list of materials to help you develop and contribute to this project. 

### What is self

https://www.educative.io/answers/what-is-self-in-python

### General approach to coding

- [Clean Code](https://thixalongmy.haugiang.gov.vn/media/1175/clean_code.pdf) (buying the book is also a good option)
- [Refactoring](http://silab.fon.bg.ac.rs/wp-content/uploads/2016/10/Refactoring-Improving-the-Design-of-Existing-Code-Addison-Wesley-Professional-1999.pdf) (buying the book is also a good option)
- [Refactoring guru](refactoring.guru), incredible resource for understanding coding patterns and how to make them better (There is also a book, could be worth investing in)
- [End of object inheritance](https://www.youtube.com/watch?v=3MNVP9-hglc) This one is hard to explain and understand until you start writing lots of code, but its worth watching (and re-watching) to understand the coding style you see in new climate models (e.g. MAGICC)
- [End of object inheritance](https://www.youtube.com/watch?v=3MNVP9-hglc) This one is hard to explain and understand until you start writing lots of code, but its worth watching (and re-watching) to understand the coding style you see in new climate models (e.g. MAGICC)
- [Composition over inheritance](https://www.youtube.com/watch?v=0mcP8ZpUR38) A nice explainer to see the principles of the above in practice
- [Dependency injection vs. inversion](https://www.youtube.com/watch?v=2ejbLVkCndI) A further explainer to see the above in practice

### Scientific software

- [Research software engineering with Python](https://merely-useful.tech/py-rse/) A little bit out of date, but a good resource for general practice and examples of developing software with python

### Numerical coding

- [Numerical recipes](http://numerical.recipes/book/book.html) (buying the book is also a good option)

### Miscellaneous

- [Basic introduction to Jupyter notebooks](https://realpython.com/jupyter-notebook-introduction/)

