"""
LUNA16 FROC Evaluation Script (Python 3)
=========================================
Computes FROC curves and performance metrics for CAD systems.
Adapted from the official LUNA16 evaluation script.

Usage:
    python froc_evaluation.py annotations.csv annotations_excluded.csv seriesuids.csv results.csv output_dir/
"""

import os
import math
import sys
from typing import List, Dict, Tuple, Optional
import numpy as np
import sklearn.metrics as skl_metrics

from NoduleFinding import NoduleFinding
from csvTools import readCSV, writeCSV

# Try to import matplotlib for plotting (optional)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. FROC plots will not be generated.")


# ============================================================================
# Evaluation Settings
# ============================================================================

bPerformBootstrapping = True
bNumberOfBootstrapSamples = 1000
bOtherNodulesAsIrrelevant = True
bConfidence = 0.95

# CSV column labels
seriesuid_label = 'seriesuid'
coordX_label = 'coordX'
coordY_label = 'coordY'
coordZ_label = 'coordZ'
diameter_mm_label = 'diameter_mm'
CADProbability_label = 'probability'

# FROC plot settings
FROC_minX = 0.125  # Minimum value of x-axis
FROC_maxX = 8      # Maximum value of x-axis
bLogPlot = True


# ============================================================================
# FROC Computation Functions
# ============================================================================

def generateBootstrapSet(scanToCandidatesDict: Dict, FROCImList: np.ndarray) -> np.ndarray:
    """Generates bootstrapped version of the candidate set."""
    imageLen = FROCImList.shape[0]
    
    # Get a random list of images using sampling with replacement
    rand_index_im = np.random.randint(imageLen, size=imageLen)
    FROCImList_rand = FROCImList[rand_index_im]
    
    # Get a new list of candidates
    candidatesExists = False
    candidates = None
    
    for im in FROCImList_rand:
        if im not in scanToCandidatesDict:
            continue
        
        if not candidatesExists:
            candidates = np.copy(scanToCandidatesDict[im])
            candidatesExists = True
        else:
            candidates = np.concatenate((candidates, scanToCandidatesDict[im]), axis=1)
    
    return candidates


def compute_mean_ci(interp_sens: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and confidence intervals from bootstrapped sensitivity values."""
    sens_mean = np.zeros((interp_sens.shape[1],), dtype='float32')
    sens_lb = np.zeros((interp_sens.shape[1],), dtype='float32')
    sens_up = np.zeros((interp_sens.shape[1],), dtype='float32')
    
    Pz = (1.0 - confidence) / 2.0
    
    for i in range(interp_sens.shape[1]):
        vec = interp_sens[:, i].copy()
        vec.sort()
        
        sens_mean[i] = np.average(vec)
        sens_lb[i] = vec[math.floor(Pz * len(vec))]
        sens_up[i] = vec[math.floor((1.0 - Pz) * len(vec))]
    
    return sens_mean, sens_lb, sens_up


def computeFROC(FROCGTList: List, FROCProbList: List, 
                totalNumberOfImages: int, excludeList: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute FROC curve from ground truth and probability lists.
    
    Returns:
        fps: False positives per scan
        sens: Sensitivity values
        thresholds: Probability thresholds
    """
    # Remove excluded candidates
    FROCGTList_local = []
    FROCProbList_local = []
    
    for i in range(len(excludeList)):
        if excludeList[i] == False:
            FROCGTList_local.append(FROCGTList[i])
            FROCProbList_local.append(FROCProbList[i])
    
    numberOfDetectedLesions = sum(FROCGTList_local)
    totalNumberOfLesions = sum(FROCGTList)
    totalNumberOfCandidates = len(FROCProbList_local)
    
    if len(FROCGTList_local) == 0 or len(set(FROCGTList_local)) == 1:
        # Edge case: no valid data or all same class
        return np.array([0]), np.array([0]), np.array([0])
    
    fpr, tpr, thresholds = skl_metrics.roc_curve(FROCGTList_local, FROCProbList_local)
    
    if sum(FROCGTList) == len(FROCGTList):
        # Handle border case when there are no false positives
        print("WARNING: This system has no false positives.")
        fps = np.zeros(len(fpr))
    else:
        fps = fpr * (totalNumberOfCandidates - numberOfDetectedLesions) / totalNumberOfImages
    
    if totalNumberOfLesions > 0:
        sens = (tpr * numberOfDetectedLesions) / totalNumberOfLesions
    else:
        sens = np.zeros(len(tpr))
    
    return fps, sens, thresholds


def computeFROC_bootstrap(FROCGTList: List, FROCProbList: List, 
                          FPDivisorList: List, FROCImList: List,
                          excludeList: List, numberOfBootstrapSamples: int = 1000,
                          confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute FROC with bootstrapping for confidence intervals."""
    
    set1 = np.concatenate(([FROCGTList], [FROCProbList], [excludeList]), axis=0)
    
    fps_lists = []
    sens_lists = []
    thresholds_lists = []
    
    FPDivisorList_np = np.asarray(FPDivisorList)
    FROCImList_np = np.asarray(FROCImList)
    
    # Make a dict with all candidates of all scans
    scanToCandidatesDict = {}
    for i in range(len(FPDivisorList_np)):
        seriesuid = FPDivisorList_np[i]
        candidate = set1[:, i:i+1]
        
        if seriesuid not in scanToCandidatesDict:
            scanToCandidatesDict[seriesuid] = np.copy(candidate)
        else:
            scanToCandidatesDict[seriesuid] = np.concatenate(
                (scanToCandidatesDict[seriesuid], candidate), axis=1
            )
    
    for i in range(numberOfBootstrapSamples):
        if i % 100 == 0:
            print(f'Computing FROC: bootstrap {i}/{numberOfBootstrapSamples}')
        
        # Generate a bootstrapped set
        btpsamp = generateBootstrapSet(scanToCandidatesDict, FROCImList_np)
        
        if btpsamp is None or btpsamp.shape[1] == 0:
            continue
            
        fps, sens, thresholds = computeFROC(
            btpsamp[0, :].tolist(), 
            btpsamp[1, :].tolist(), 
            len(FROCImList_np), 
            btpsamp[2, :].tolist()
        )
        
        fps_lists.append(fps)
        sens_lists.append(sens)
        thresholds_lists.append(thresholds)
    
    # Compute statistics
    all_fps = np.linspace(FROC_minX, FROC_maxX, num=10000)
    
    # Interpolate all FROC curves at these points
    interp_sens = np.zeros((len(fps_lists), len(all_fps)), dtype='float32')
    for i in range(len(fps_lists)):
        interp_sens[i, :] = np.interp(all_fps, fps_lists[i], sens_lists[i])
    
    # Compute mean and CI
    sens_mean, sens_lb, sens_up = compute_mean_ci(interp_sens, confidence=confidence)
    
    return all_fps, sens_mean, sens_lb, sens_up


# ============================================================================
# Helper Functions
# ============================================================================

def getNodule(annotation: List[str], header: List[str], state: str = "") -> NoduleFinding:
    """Create a NoduleFinding object from CSV annotation."""
    nodule = NoduleFinding()
    nodule.coordX = float(annotation[header.index(coordX_label)])
    nodule.coordY = float(annotation[header.index(coordY_label)])
    nodule.coordZ = float(annotation[header.index(coordZ_label)])
    
    if diameter_mm_label in header:
        nodule.diameter_mm = float(annotation[header.index(diameter_mm_label)])
    
    if CADProbability_label in header:
        nodule.CADprobability = float(annotation[header.index(CADProbability_label)])
    
    if state:
        nodule.state = state
    
    return nodule


def collectNoduleAnnotations(annotations: List, annotations_excluded: List, 
                             seriesUIDs: List[str]) -> Dict[str, List[NoduleFinding]]:
    """Collect all nodule annotations into a dictionary keyed by series UID."""
    allNodules = {}
    noduleCount = 0
    noduleCountTotal = 0
    
    for seriesuid in seriesUIDs:
        nodules = []
        numberOfIncludedNodules = 0
        
        # Add included findings
        header = annotations[0]
        for annotation in annotations[1:]:
            nodule_seriesuid = annotation[header.index(seriesuid_label)]
            
            if seriesuid == nodule_seriesuid:
                nodule = getNodule(annotation, header, state="Included")
                nodules.append(nodule)
                numberOfIncludedNodules += 1
        
        # Add excluded findings
        header = annotations_excluded[0]
        for annotation in annotations_excluded[1:]:
            nodule_seriesuid = annotation[header.index(seriesuid_label)]
            
            if seriesuid == nodule_seriesuid:
                nodule = getNodule(annotation, header, state="Excluded")
                nodules.append(nodule)
        
        allNodules[seriesuid] = nodules
        noduleCount += numberOfIncludedNodules
        noduleCountTotal += len(nodules)
    
    print(f'Total included nodule annotations: {noduleCount}')
    print(f'Total nodule annotations: {noduleCountTotal}')
    
    return allNodules


# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluateCAD(seriesUIDs: List[str], results_filename: str, outputDir: str,
                allNodules: Dict[str, List[NoduleFinding]], CADSystemName: str,
                maxNumberOfCADMarks: int = -1, performBootstrapping: bool = False,
                numberOfBootstrapSamples: int = 1000, confidence: float = 0.95) -> Dict:
    """
    Evaluate a CAD algorithm and compute FROC metrics.
    
    Returns:
        Dictionary with evaluation results and FROC data
    """
    os.makedirs(outputDir, exist_ok=True)
    
    nodOutputfile = open(os.path.join(outputDir, 'CADAnalysis.txt'), 'w')
    nodOutputfile.write("\n")
    nodOutputfile.write(("=" * 60) + "\n")
    nodOutputfile.write(f"CAD Analysis: {CADSystemName}\n")
    nodOutputfile.write(("=" * 60) + "\n\n")
    
    results = readCSV(results_filename)
    allCandsCAD = {}
    
    for seriesuid in seriesUIDs:
        # Collect candidates from result file
        nodules = {}
        header = results[0]
        
        i = 0
        for result in results[1:]:
            nodule_seriesuid = result[header.index(seriesuid_label)]
            
            if seriesuid == nodule_seriesuid:
                nodule = getNodule(result, header)
                nodule.candidateID = i
                nodules[nodule.candidateID] = nodule
                i += 1
        
        # Apply max CAD marks limit if specified
        if maxNumberOfCADMarks > 0 and len(nodules) > maxNumberOfCADMarks:
            probs = [(key, float(n.CADprobability)) for key, n in nodules.items()]
            probs.sort(key=lambda x: x[1], reverse=True)
            
            nodules = {k: nodules[k] for k, _ in probs[:maxNumberOfCADMarks]}
        
        allCandsCAD[seriesuid] = nodules
    
    # Initialize tracking variables
    candTPs = 0
    candFPs = 0
    candFNs = 0
    totalNumberOfCands = 0
    totalNumberOfNodules = 0
    doubleCandidatesIgnored = 0
    irrelevantCandidates = 0
    minProbValue = -1e10
    
    FROCGTList = []
    FROCProbList = []
    FPDivisorList = []
    excludeList = []
    
    # Loop over cases
    for seriesuid in seriesUIDs:
        candidates = allCandsCAD.get(seriesuid, {})
        totalNumberOfCands += len(candidates)
        candidates2 = candidates.copy()
        
        noduleAnnots = allNodules.get(seriesuid, [])
        
        for noduleAnnot in noduleAnnots:
            if noduleAnnot.state == "Included":
                totalNumberOfNodules += 1
            
            x = float(noduleAnnot.coordX)
            y = float(noduleAnnot.coordY)
            z = float(noduleAnnot.coordZ)
            
            diameter = float(noduleAnnot.diameter_mm)
            if diameter < 0.0:
                diameter = 10.0
            radiusSquared = pow((diameter / 2.0), 2.0)
            
            found = False
            noduleMatches = []
            
            for key, candidate in list(candidates.items()):
                x2 = float(candidate.coordX)
                y2 = float(candidate.coordY)
                z2 = float(candidate.coordZ)
                dist = pow(x - x2, 2.) + pow(y - y2, 2.) + pow(z - z2, 2.)
                
                if dist < radiusSquared:
                    if noduleAnnot.state == "Included":
                        found = True
                        noduleMatches.append(candidate)
                        if key in candidates2:
                            del candidates2[key]
                    elif noduleAnnot.state == "Excluded":
                        if bOtherNodulesAsIrrelevant and key in candidates2:
                            irrelevantCandidates += 1
                            del candidates2[key]
            
            if len(noduleMatches) > 1:
                doubleCandidatesIgnored += (len(noduleMatches) - 1)
            
            if noduleAnnot.state == "Included":
                if found:
                    maxProb = max(float(c.CADprobability) for c in noduleMatches)
                    FROCGTList.append(1.0)
                    FROCProbList.append(maxProb)
                    FPDivisorList.append(seriesuid)
                    excludeList.append(False)
                    candTPs += 1
                else:
                    candFNs += 1
                    FROCGTList.append(1.0)
                    FROCProbList.append(minProbValue)
                    FPDivisorList.append(seriesuid)
                    excludeList.append(True)
        
        # Add false positives
        for key, candidate in candidates2.items():
            candFPs += 1
            FROCGTList.append(0.0)
            FROCProbList.append(float(candidate.CADprobability))
            FPDivisorList.append(seriesuid)
            excludeList.append(False)
    
    # Write results
    sensitivity = float(candTPs) / float(totalNumberOfNodules) if totalNumberOfNodules > 0 else 0.0
    avg_fps = float(candFPs) / float(len(seriesUIDs)) if len(seriesUIDs) > 0 else 0.0
    
    nodOutputfile.write("Candidate detection results:\n")
    nodOutputfile.write(f"    True positives: {candTPs}\n")
    nodOutputfile.write(f"    False positives: {candFPs}\n")
    nodOutputfile.write(f"    False negatives: {candFNs}\n")
    nodOutputfile.write(f"    Total candidates: {totalNumberOfCands}\n")
    nodOutputfile.write(f"    Total nodules: {totalNumberOfNodules}\n")
    nodOutputfile.write(f"    Ignored (excluded nodules): {irrelevantCandidates}\n")
    nodOutputfile.write(f"    Ignored (double detections): {doubleCandidatesIgnored}\n")
    nodOutputfile.write(f"    Sensitivity: {sensitivity:.6f}\n")
    nodOutputfile.write(f"    Avg FPs per scan: {avg_fps:.6f}\n")
    nodOutputfile.close()
    
    # Compute FROC
    fps, sens, thresholds = computeFROC(FROCGTList, FROCProbList, len(seriesUIDs), excludeList)
    
    fps_bs_itp = sens_bs_mean = sens_bs_lb = sens_bs_up = None
    
    if performBootstrapping and len(FROCGTList) > 0:
        fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up = computeFROC_bootstrap(
            FROCGTList, FROCProbList, FPDivisorList, seriesUIDs, excludeList,
            numberOfBootstrapSamples=numberOfBootstrapSamples, confidence=confidence
        )
    
    # Write FROC data to CSV
    with open(os.path.join(outputDir, f"froc_{CADSystemName}.csv"), 'w') as f:
        f.write("fps,sensitivity,threshold\n")
        for i in range(len(sens)):
            f.write(f"{fps[i]:.9f},{sens[i]:.9f},{thresholds[i]:.9f}\n")
    
    # Interpolate at standard FP rates
    standard_fps = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)
    sens_itp = np.interp(fps_itp, fps, sens)
    
    # Get sensitivity at standard operating points
    sens_at_fps = {}
    for fp_rate in standard_fps:
        idx = np.argmin(np.abs(fps_itp - fp_rate))
        sens_at_fps[fp_rate] = float(sens_itp[idx])
    
    # Create FROC plot if matplotlib available
    if HAS_MATPLOTLIB and totalNumberOfNodules > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(fps_itp, sens_itp, color='#2563EB', label=CADSystemName, lw=2)
        
        if performBootstrapping and fps_bs_itp is not None:
            ax.plot(fps_bs_itp, sens_bs_mean, color='#2563EB', ls='--', alpha=0.7)
            ax.fill_between(fps_bs_itp, sens_bs_lb, sens_bs_up, 
                           facecolor='#2563EB', alpha=0.1, label='95% CI')
        
        ax.set_xlim(FROC_minX, FROC_maxX)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Average number of false positives per scan', fontsize=12)
        ax.set_ylabel('Sensitivity', fontsize=12)
        ax.legend(loc='lower right')
        ax.set_title(f'FROC Performance - {CADSystemName}', fontsize=14)
        
        if bLogPlot:
            ax.set_xscale('log', base=2)
            ax.xaxis.set_major_formatter(FixedFormatter(['0.125', '0.25', '0.5', '1', '2', '4', '8']))
        
        ax.set_xticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outputDir, f"froc_{CADSystemName}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Return results
    return {
        'sensitivity': sensitivity,
        'specificity': 1.0 - (candFPs / (candFPs + totalNumberOfNodules)) if (candFPs + totalNumberOfNodules) > 0 else 0.0,
        'true_positives': candTPs,
        'false_positives': candFPs,
        'false_negatives': candFNs,
        'total_nodules': totalNumberOfNodules,
        'total_candidates': totalNumberOfCands,
        'avg_fps_per_scan': avg_fps,
        'sensitivity_at_fps': sens_at_fps,
        'froc_fps': fps.tolist() if isinstance(fps, np.ndarray) else fps,
        'froc_sens': sens.tolist() if isinstance(sens, np.ndarray) else sens,
        'froc_fps_bootstrap': fps_bs_itp.tolist() if fps_bs_itp is not None else None,
        'froc_sens_mean': sens_bs_mean.tolist() if sens_bs_mean is not None else None,
        'froc_sens_lb': sens_bs_lb.tolist() if sens_bs_lb is not None else None,
        'froc_sens_up': sens_bs_up.tolist() if sens_bs_up is not None else None
    }


def collect(annotations_filename: str, annotations_excluded_filename: str, 
            seriesuids_filename: str) -> Tuple[Dict, List[str]]:
    """Load annotations and series UIDs from CSV files."""
    annotations = readCSV(annotations_filename)
    annotations_excluded = readCSV(annotations_excluded_filename)
    seriesUIDs_csv = readCSV(seriesuids_filename)
    
    seriesUIDs = [row[0] for row in seriesUIDs_csv if row]
    
    allNodules = collectNoduleAnnotations(annotations, annotations_excluded, seriesUIDs)
    
    return allNodules, seriesUIDs


def noduleCADEvaluation(annotations_filename: str, annotations_excluded_filename: str,
                        seriesuids_filename: str, results_filename: str, 
                        outputDir: str) -> Dict:
    """
    Main function to load annotations and evaluate a CAD algorithm.
    
    Args:
        annotations_filename: CSV with nodule annotations
        annotations_excluded_filename: CSV with excluded annotations
        seriesuids_filename: CSV with series UIDs to evaluate
        results_filename: CSV with CAD system predictions
        outputDir: Output directory for results
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Loading annotations from: {annotations_filename}")
    
    allNodules, seriesUIDs = collect(
        annotations_filename, 
        annotations_excluded_filename, 
        seriesuids_filename
    )
    
    results = evaluateCAD(
        seriesUIDs, results_filename, outputDir, allNodules,
        os.path.splitext(os.path.basename(results_filename))[0],
        maxNumberOfCADMarks=100,
        performBootstrapping=bPerformBootstrapping,
        numberOfBootstrapSamples=bNumberOfBootstrapSamples,
        confidence=bConfidence
    )
    
    return results


# ============================================================================
# Command Line Interface
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: python froc_evaluation.py annotations.csv annotations_excluded.csv "
              "seriesuids.csv results.csv output_dir/")
        sys.exit(1)
    
    annotations_filename = sys.argv[1]
    annotations_excluded_filename = sys.argv[2]
    seriesuids_filename = sys.argv[3]
    results_filename = sys.argv[4]
    outputDir = sys.argv[5]
    
    results = noduleCADEvaluation(
        annotations_filename, 
        annotations_excluded_filename,
        seriesuids_filename, 
        results_filename, 
        outputDir
    )
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Sensitivity: {results['sensitivity']:.4f}")
    print(f"Avg FPs/scan: {results['avg_fps_per_scan']:.4f}")
    print("\nSensitivity at operating points:")
    for fp, sens in results['sensitivity_at_fps'].items():
        print(f"  @ {fp} FP/scan: {sens:.4f}")
    print("=" * 60)
