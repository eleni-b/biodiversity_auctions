# Budget-Feasible Market Design for Biodiversity Conservation

This repository contains code and data evaluating the performance of various mechanisms for conservation auctions. We examine novel and established auction algorithms and assess their suitability for the purpose of biodiversity conservation from the perspective of a budget-constrained authority.



The corresponding publication can be found in: https://aisel.aisnet.org/cgi/viewcontent.cgi?article=1043&context=wi2023

**Authors:** Eleni Batziou, Martin Bichler


## Usage

Valuations for land parcels and farmer costs used in the auction are generated on the fly based on the valuation function of choice. For this, the user is required to insert a random seed. The experimental code, implementing and evaluating the auction algorithms on different value function models is contained entirely within run_all.py. To invoke the script, insert a random seed and simply run:


`python run_all.py <random seed>`
