# VLSI

VLSI (Very Large Scale Integration) refers to the trend of integrating circuits into silicon chips. A typical example is the smartphone. The modern trend of shrinking transistor sizes, allowing engineers to fit more and more transistors into the same area of silicon, has pushed the integration of more and more functions of cellphone circuitry into a single silicon die (i.e. plate). This enabled the modern cellphone to mature into a powerful tool that shrank from the size of a large brick-sized unit to a device small enough to comfortably carry in a pocket or purse, with a video camera, touchscreen, and other advanced features.
In this project, we developed four different approaches for the VLSI problem: in particular, we adopted Constraint Programing (CP), Propositional Satisfiability (SAT), Satisfiability Modulo Theories (SMT) and Mixed Integer Linear Programming (MIP) in ***jupyter notebook***.

<br>

The project presents an organizational structure composed of four main folders: CP, SMT, SAT, and MIP. Each of these folders contains three subfolders, namely:
- ```src```: containing the source code of the specific model;
- ```images```: containing the images of the results of the specific model;
- ```out```: containing the results of the specific model in .txt format.
<br>

There is also a folder ```utils```, in addition to the ones listed, which includes further important elements for the project. This folder contains the instances used as input for the models, the various sets of data where the results have been saved, and a ```utils.py``` file that is used to process the results and create the images.
Additionally, there is a notebook called ```results.ipynb``` containing the results of the 4 models for each instance.
