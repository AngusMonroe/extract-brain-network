# Step by Step procedure to extract connectivity matrices - for a single subject
#Sample TaoWu dataset (INCLUDE SUBJECT NAME) is included in demo folder
#The 

STEP 1 - Convert DICOM to BIDS formatted dataset. 
The sample data/subject provided is TaoWu dataset. TaoWu dataset are BIDS formatted dataset. Hence the processing step of DICOM to BIDS conversion is not included for Demo Purpose. 

STEP 2 - Preprocess data using fmriprep.
1. Install fmriprep - INCLUDE LINK
2. Single line command to run the subject is as follows "fmriprep-docker /input_folder_location/Demo/sample_taowu /output_folder_location/Demo/sample_taowu/derivatives participant --participant-label control002S0413 --skip-bids-validation --stop-on-first-crash --md-only-boilerplate --fs-no-reconall --output-spaces MNI152NLin2009cAsym:res-2 --fs-license-file /license_location/Freesurfer_license/license.txt --ignore slicetiming"

STEP 3 - Extract Connectivity matrices





