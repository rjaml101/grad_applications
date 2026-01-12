# Euler Elastica

A closed 3D Euler Elastic curve can be interpreted as **minimizing total bending energy, constrained by constant total torsion and constant length**. However, there exists a simpler characterization: 3D Euler Elastica can also be defined by **minimizing length, constrained by a constant “projected” area vector and “revolved” volume vector**. This is an easier way to generate Euler Elastic curves, and it’s also easier to discretize.  

The goal was to come up with the proper numerical/iterative method to generate discrete closed 3D Euler Elastic curves, by starting with randomly initialized points in 3D space, and using a simple length minimization method while constraining the Area vector and Volume vector.  

- “Area vector” refers to the vector of three projected areas of the 3D closed curve onto the 3 coordinate planes (xy, yz, xz).  
- “Volume vector” refers to the vector containing the three volumes for the toruses created by revolving the closed 3D curve about the 3 coordinate axes (x, y, z).  

## Algorithms Used / Considered

- Helped with discretizations, derivations and proof-of-concept implementations in Python and MATLAB, for the Augmented Lagrangian (AL) method.  
- Came up with a simple scalene/isosceles algorithm for length minimization under 2D scalar area constraint. Though it is simpler than the default Augmented Lagrangian method, it unfortunately didn’t generalize beyond the planar case, so we went back to the Augmented Lagrangian version **[Attached "Note on 2D Algorithm.pdf" to this folder]**.  

## Result

The code now generates closed 3D Euler Elastic curves based on length minimizer with area vector and volume vector constraints (using an Augmented Lagrangian), as opposed to the more complex method of minimizing total bending energy with length and total torsion constraints.  

**EXAMPLES OF GENERATED EULER ELASTIC CURVES WITH VARIOUS AREA/VOLUME VECTOR CONSTRAINTS**

CIRCLE: 

A = [5,5,5]; V = [0,0,0]

<img width="349" height="287" alt="A5_V0_circle" src="https://github.com/user-attachments/assets/67585f56-5853-4827-8293-a53dba6ef3dc" />


BUCKLED LOOP:

A = [5,5,5]; V = [5,5,5]

<img width="402" height="276" alt="A5_V5_buckled_loop" src="https://github.com/user-attachments/assets/44c4350f-9566-422a-ae36-92e8e4e0113b" />



FIGURE EIGHT:

A = [0,0,0]; V = [5,5,5]

<img width="431" height="186" alt="A0_V5_figure_eight" src="https://github.com/user-attachments/assets/7c681531-98fa-4172-9409-7ff40c0a47a7" />




## Extensions

- We can also try to extend this to open curves.  
- This also has implications for splines in CAD. 
