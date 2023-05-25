---
title: 'pyRISM: A Python package for classical fluid theory'
tags:
  - Python
  - solvation
  - solvation free energy
  - integral equation
  - chemistry
authors:
  - name: Abdullah Ahmad
    orcid: 0000-0002-6315-749X
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: David S. Palmer
    orcid: 0000-0003-4356-9144
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
affiliations:
 - name: Department of Pure and Applied Chemistry, University of Strathclyde, Thomas Graham Building, 295 Cathedral Street, Glasgow G1 1XL, United Kingdom
   index: 1
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Solvation---and more generally the liquid state---are important fields of study in chemistry.
A majority of reactions occur in solution.
As a result, it is pertinent that liquids and solvation can be modelled with physics and computational models.
A conceptually straightforward approach to modelling the liquid state is via molecular dynamics (MD). 
The solvent is represented atomistically and placed in the simulation cell around a solute, and the dynamics are simulated using the equations of motion.
Thermodynamic properties can be calculated from the resulting output.
The issue with explicitly modelling the solvent in this manner is the computational expense required to obtain statistically meaningful sampling.
Simulations are heavily dependent on system size such that computational time can range anywhere from hours to weeks.
A faster method is to represent the solvent as a continuous medium---an implicit solvent---by statistically averaging over all the degrees of freedom of the solvent.
The medium is parametrised using the dielectric constant of the solvent.
This approach reduces the computational expense, but the amount of information obtained from an implicit solvation calculation is limited in comparison to explicit solvation [@jensenIntroductionComputationalChemistry2017;@cramerEssentialsComputationalChemistry2013].
A third purely statistical-mechanical approach is based on density distribution functions that are used to represent solvent density around a solute.
Still somewhat implicit by nature, the computational expense sits between explicit and continuum models, yet the density distributions yield information about the structure of the liquid [@hansen1990theory].
The origins of this theory lies in the work of Ornstein et al. [@ornstein1914accidental] in which correlations between particles are decomposed into direct and indirect components---$c(r)$ and $\gamma(r)$ respectively.
In order to extend this approach to molecular liquids, the reference interaction site model (RISM) was developed by Chandler et al. [@chandlerOptimizedClusterExpansions1972] (sometimes called the site-site Ornstein-Zernike (SSOZ) equation) as a way to approximate the molecular Ornstein-Zernike (MOZ) equation, turning a 6D equation into a set of 1D equations.
This significantly simplifies the calculation by averaging over the orientations of a molecule.
The one dimensional RISM equation (1D-RISM) was modified to better model charged systems, resulting in the extended reference interaction site model (XRISM).
Further development took place to address the RISM equation not accurately calculating the dielectric constant.
The dielectric constant is instead used as input into the equation, resulting in the dielectrically consistent reference interaction site model (DRISM).
1D-RISM and its variations can be used to understand solvent structure in solution, as well as calculating solvation free energies.
There are a few 1D-RISM implementations available.
The AMBER implementation [@luchkoThreeDimensionalMolecularTheory2010] calculates solvent-solvent interactions, but lacks a solute-solvent implementation.
RISM-MOL developed by Sergiievsky et al. [@sergiievskyiMultigridSolverReference2011] provides a solute-solvent implementation, but only works for aqueous solvents at 298K.
pyRISM includes both solving for solute-solvent interaction, as well the ability to solve the RISM equations at various temperatures.
In this paper, we present the pyRISM [@ahmad_pyrism_2023] software package and discuss the underlying theory, implementation details, differences with available packages and development principles as well as presenting a few preliminary results to showcase its current capabilities.

# Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

Results were obtained using the ARCHIE-WeSt High Performance Computer (www.archie-west.ac.uk) based at the University of Strathclyde.
D.S.P and A.A thank EPSRC for funding via a PhD studentship for A.A.

# References