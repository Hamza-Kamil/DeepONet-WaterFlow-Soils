# Physics-informed neural operators for efficient modeling of infiltration in porous media

This repository contains the code implementations used for the research paper titled "Physics-informed neural operators for efficient modeling of water flow in unsaturated soils."

## Authors

- **Hamza Kamil** (a,b)
- **Azzeddine Soulaïmani** (b)
- **Abdelaziz Beljadid** (a,c)

## Affiliation  

- **Mohammed VI Polytechnic University, Morocco** (a)
- **École de technologie supérieure, Canada** (b)
- **University of Ottawa, Canada** (c)

## Key Points

(a) For a detailed understanding of the numerical setup for each case, please refer to the accompanying research paper.

(b) The DeepONet model is implemented using the JAX library.


## Getting Started

To use the code, follow these steps:

1. Use Google Colab or set up the code on your local machine.
2. Install required libraries (including the latest version of JAX)
3. All the necesseray libraries should be carefuly installed.
4. The finite element reference solution is given as well.
5. Explore the code files to understand the implementation details.

## Key Features

- DeepONet model implementation using JAX library.
- Finite element reference solution provided.
- Detailed numerical setup for scenario (b) (refer to the paper).

## Results Visualization

| Water Flow Simulation | 2D Mesh |
|:----------------------:|:-------------------:|
| <img src="DeepONet_Codes/2Dinfiltration.gif" width="400" alt="Results GIF"> | <img src="DeepONet_Codes/2Dmesh.png" width="400" alt="2D Mesh"> |

Left: The GIF demonstrates water infiltration in unsaturated soils as modeled by our physics-informed neural operator (DeepONet).

Right: 2D unstructured mesh used in the finite element simulation.

## Note

This repository is intended to serve as a resource for researchers and practitioners interested in advanced solvers for solving partial differential equations related to water flow and multiple solute transport in unsaturated soils.

Contributions and improvements are welcome. Feel free to submit pull requests or open issues for any questions or suggestions.

## Contact Information

For inquiries, please reach out to:

- **Hamza Kamil** (hamza.kamil@um6p.ma, hamza.kamil.1@ens.etsmtl.ca)

## Citation

If you find this work useful for your research, please consider citing the accompanying paper.

Thank you for your interest in our research!
