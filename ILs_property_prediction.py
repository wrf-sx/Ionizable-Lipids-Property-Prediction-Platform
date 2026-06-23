import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
import os

# Set network URL
#NETWORK_URL = "http://ILspropertyprediction.computpharm.org"

# Define SMILES standardization function
def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        standardized_mol = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return standardized_mol
    except:
        return None

# Calculate ECFP fingerprint
def calculate_ecfp(smiles, radius=3, bits=1024, chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=bits, useChirality=chirality
    )
    return np.array(fp)

# Add formulation features
def add_formulation_features(ecfp, features_dict):
    features = list(features_dict.values())
    return np.concatenate([ecfp, features])

# Save features to CSV
def save_features_to_csv(features, filename):
    df = pd.DataFrame([features])
    df.to_csv(filename, index=False, header=False)
    return filename

# Main function
def main():
    st.title("ILs Property Prediction Platform")
    #st.markdown(f"**Network URL**: {NETWORK_URL}")
    st.markdown("Input SMILES string of ILs for property prediction")
    
    # User input
    smiles_input = st.text_input("Enter SMILES string of ILs:", "")
    
    if st.button("Predict"):
        if not smiles_input:
            st.warning("Please enter a valid SMILES string")
            return
            
        # Standardize SMILES
        standardized_smiles = standardize_smiles(smiles_input)
        if standardized_smiles is None:
            st.error("Unable to parse SMILES string. Please check your input.")
            return
            
        st.success(f"Standardized SMILES: {standardized_smiles}")
        
        # Calculate ECFP
        ecfp = calculate_ecfp(standardized_smiles)
        if ecfp is None:
            st.error("Failed to calculate ECFP fingerprint")
            return
            
        # Define formulation features
        pka_formulation = {
            "LipRat_IL": 0.5, "LipRat_Cho": 0.385, "LipRat_Help": 0.1, 
            "LipRat_PEG": 0.015, "Helper_DPhyPE": 0, "Helper_DSPC": 1,
            "PEG_type_DMG-PEG": 1, "PEG_type_PEG-DMA": 0, "Protein_hEPO": 0,
            "Protein_luciferase": 1, "N/P ratio": 6, "Mass ratio": -1
        }
        
        eff_formulation = {
            'LipRat_IL': 0.5, 'LipRat_Cho': 0.385, 'LipRat_Help': 0.1,
            'LipRat_PEG': 0.015, 'Helper_DSPC': 1, 'PEG_DMG-PEG2000': 1,
            'PEG_DMPE-PEG2000': 0, 'PEG_PEG-DMA': 0, 'PEG_PEG2000-C-DMA': 0,
            'Protein_eGEP': 0, 'Protein_hEPO': 0, 'Protein_luciferase': 1,
            'Mice_BALB/c': 1, 'Mice_C57BL/6': 0, 'Mice_CD-1': 0, 'Mice_ICR': 0,
            'Time': 4, 'N/P ratio': 6, 'Mass ratio': -1
        }
        
        # Build feature sets
        pka_features = add_formulation_features(ecfp, pka_formulation)
        eff_features = add_formulation_features(ecfp, eff_formulation)
        mem_features = eff_features.copy()  # Use same feature set
        
        # Save feature files
        pka_file = save_features_to_csv(pka_features, "pka_features.csv")
        eff_file = save_features_to_csv(eff_features, "eff_features.csv")
        mem_file = save_features_to_csv(mem_features, "mem_features.csv")
        
        # Load models
        try:
            pka_model = joblib.load("pKa_ECFP6_LGB.pkl")
            eff_model = joblib.load("ecfp6_Efficiency_LGB.pkl")
            
            # Special handling for TabPFN model
            mem_model = joblib.load("LightGBM_model.pkl")
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            st.error("Ensure all model files exist and required libraries are installed")
            st.error("Install required packages with: pip install tabpfn rdkit-pypi scikit-learn")
            return
            
        # Make predictions
        try:
            pka_pred = pka_model.predict(pd.read_csv(pka_file, header=None))[0]
            eff_pred = eff_model.predict(pd.read_csv(eff_file, header=None))[0]
            mem_pred = mem_model.predict(pd.read_csv(mem_file, header=None))[0]
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return
            
        # Interpret prediction results
        pka_result = {
            0: "apparent pKa ≤ 6",
            1: "6 < apparent pKa < 7",
            2: "apparent pKa ≥ 7"
        }[pka_pred]
        
        eff_result = {
            0: "Deliver mRNA ability < MC3",
            1: "Deliver mRNA ability ≥ MC3"
        }[eff_pred]
        
        mem_result = {
            0: "Endosomal membrane fusion ability < SM102",
            1: "Endosomal membrane fusion ability ≥ SM102"
        }[mem_pred]
        
        # Display results
        st.subheader("Prediction Results:")
        st.markdown(f"**Apparent pKa Prediction**: {pka_result}")
        st.markdown(f"**Deliver mRNA Ability Prediction**: {eff_result}")
        st.markdown(f"**Endosomal Escape Ability Prediction**: {mem_result}")
        
        # Clean up temporary files
        try:
            os.remove(pka_file)
            os.remove(eff_file)
            os.remove(mem_file)
        except:
            pass

if __name__ == "__main__":
    main()
