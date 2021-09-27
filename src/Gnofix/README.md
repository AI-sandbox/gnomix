# Gnofix 

![Visualization of the process](figures/XGFix.gif)

## Motivation

Accurate phasing of genomic data is crucial for human demographic modeling and identity-by-descent analyses. It has been shown that leveraging information about an individual’s genomic ancestry improves performance of current phasing algorithms. Gnofix is a method that uses local ancestry inference (LAI) to do exactly that.

![Local Ancestry for Phasing Error Correction](figures/laipec_resized.png)
Sequenced haplotypes phased with a phasing software (left). LAI used to label haplotypes with ancestry predictions and phasing errors become evident (center). Phasing error correction using LAI is applied to correct phasing errors (right).

## Method

Gnofix uses a trained smoother from Gnomix to estimate the probability of a given sequence being from the distribution of a human haplotype (correctly phased) and leverages those estimates to iteratively switch parts along the haplotypes to ultimately find the most probable one. 

![Gnofix Diagram](figures/diagram_resized.png)
- (a) Maternal and paternal haplotypes at a given iteration and a given step of Gnofix. Dashed box marks the scope. 
- (b) Ancestry probabilities at windows in scope are extracted.
- (c) Windows in scope are permuted (to form possible fixes).
- (d)-(e) Permutations are passed through the smoother to obtain predictions and their confidence.
- (f) Permutation with highest confidence is chosen.
- (g) Windows in scope are returned to the haplotypes, preserving the scope’s right boundary (i.e. if rightmost window switched between haplotypes, entire 
right part switches too), before moving the scope one window to the right for next step.

## Result

Below is a visualization of the process for one - in this case the first - iteration of the algorithm applied on real data. The two haplotypes are from a simulated Latino individual with the red color corresponding to African segment, blue to the European segment and green to the indigenous American segment.


![Visualization of the process](figures/XGFix.gif)


Gnofix starts off with the maternal and paternal haplotypes phased from Beagle, a standard phasing softwere. The ancestry is inferred by Gnomix and phasing errors become evident as ancestries switch haplotypes at exactly the same position repeatedly. At each step Gnofix performs the procedure described above. Once the complete iteration is over, one can see how the ancestry predictions are almost uniform and resemble a chromosome pair with contiguous segments inherited from the 3 different ancestries, rather than ancestries switching simultaneously, and in concert, at the same position on both haplotypes, which is highly unlikely to occur outside of phasing errors.
