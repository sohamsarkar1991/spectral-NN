Dynamic functional principal component
S. Hörmann, ?. Kidzinski and M. Hallin
J. R. Statist. Soc. B, 77 (2015), 319 -- 348


1. CONTENTS

src directory contains subdirectories:
    data - with PM10 dataset.
    lib - with a set of R functions for computing the Dynamic Principal Components
    figures - here all the figures are generated (initially empty)

It also contains R files:
    library.R - loads all necessery files from lib directory
    loaddata.R - transforms raw data into a functional time series
    pm10.R - roal data experiment (Chapter 5)
    simulation.R - simulation study (Chapter 6)


2. PM10 DATASET

File 'data/pm10.txt' contains the data used in the experiment described in Chapter 5. First two columns are dates and hours of measurments. Column 3 is the number of observations. Column 4 is the concentration of particulate matter, known as PM10 in Graz, Austria from October 1, 2010 through March 31, 2011.


3. REQUIREMENTS

In order to run our software one needs
   - R environment (http://www.r-project.org/)
   - 'fda' library for R (http://cran.r-project.org/web/packages/fda/index.html).


4. USAGE

Start R in the directory 'src' and type

    source('pm10.R')

This script computes the values from the paper and generates the figures in the 'figures' directory.
For the simulation study run

    source('simulation.R')

This generates the data for the Table 1. Note that the simulation study was designed for a cluster computers, thus it is not suitable for a single PC without decreasing the parameters (like the number of repetitions and the dimension).


Siegfried Hörmann
Department of Mathematics,
Université libre de Bruxelles
CP 210
Boulevard du Triomphe
Belgium

E-mail: shormann@ulb.ac.be