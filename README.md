# Numerical Integration on GPUs 

This repository contains code developed in support of Fermi
National Accelerator Laboratory's LDRD project 2020-050,
"Numerical Integration on GPUs".

## Obtaining the code

This project uses *git submodules* to obtain several other codebases.
The easiest way to obtain *all* the code needed is to use a special
form of the *git clone* command:

```
git clone --recurse-submodules ...
```

If you clone a new submodule repository into this one, after doing
the clone, remember to update our list of submodules:

```
git submodule update --init --recursive
```
