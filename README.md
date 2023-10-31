# DAESIM

## Description

This repository contains the python implementation of the Dynamic Agro-Ecosystem SIMulator (DAESIM) model. 

References: Taghikhah et al. (2022) https://doi.org/10.1016/j.ecolmodel.2022.109930

## Installation

Download the DAESim git repository onto your local machine by clicking on the green "Code" button and following your download method of choice. Once you have the DAESim repository on your local machine, you must install it by following the instructions below.

Before installation of DAESim, you will need to have the following things installed:
- pip: A Python package management tool. It is the recommended tool for installing and managing Python packages. 
- Anaconda (Conda): A cross-platform, package and environment management system. 
Conda and pip do a lot of similar things. Here, we use Conda to install Python package dependencies as well as to manage the virtual environment, while we use pip to install the DAESim package source code. 

### Installing tools on Mac

Text todo

## Getting started

Make your conda environment called `daesim` using the environment.yml file and the command:

$ conda env create --name daesim --file environment.yml

Then, you can activate your new environment with the command:

$ conda activate daesim

Now you have created the conda virtual environment to work in, we must install the package. To install the package use the command:

$ pip install -e .

Hopefully everything installed successfully. 

You can now run the code or open a jupyter notebook to start testing. To open a jupyter notebook run `jupyter notebook` to open a jupyter server. From there, go into the notebooks directory and work through the examples.

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

## General guidance on working in this repository

### Repository guiding principles

Text todo: Outline how to use this repository. e.g. /data/ should only contain a test set of forcing data, do not commit large datasets to it. Similarly, do not commit anything to /results/. Keep the repo clean and tidy. 

### Branch etiquette

In general, don't push to other people's branches (things can get weird if two people work on the same branch at once).
Instead, check out the branch and then create your own branch in case you want to make any commits (see "Checking out a branch related to a merge request" below).
Then, to add your changes, make a merge request from your own branch back into the original branch.

## Support

If you need support, the first place to go is the [issue tracker] (todo: link the repository / issues).
From there, you can tag other model users to ask for help.
As a second step, reach out directly to your collaborators.

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

## Reading and viewing

Not all of these will be neccesary for everyone, but a helpful list

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

### Big climate model output and handling netCDF files

- [Software Carpentry's introduction to Python for atmosphere and ocean scientists](https://carpentries-lab.github.io/python-aos-lesson/), particularly if you think you're going to be working with netCDF files
- [CMIP6 in the cloud](https://medium.com/pangeo/cmip6-in-the-cloud-five-ways-96b177abe396), particularly if you're going to be dealing with CMIP data a lot (although speak with someone about how far you need to go, it's unlikely that every step will be relevant)

### Miscellaneous

- [Basic introduction to Jupyter notebooks](https://realpython.com/jupyter-notebook-introduction/)

## Recommended tools

- An IDE (e.g. pycharm) or more light-weight editor (e.g. Sublime)
- If on Mac, homebrew
- Git
- Make
- Slack
- gfortran
- cmake
