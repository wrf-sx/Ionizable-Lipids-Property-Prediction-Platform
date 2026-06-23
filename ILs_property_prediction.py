import os

import streamlit as st
import pandas as pd
import numpy as np
import joblib

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ModuleNotFoundError:
    Chem = None
    AllChem = None


# Define SMILES standardization function
def standardize_smiles(smiles):
    if Chem is None:
        st.error(
            "RDKit is not installed. Please add 'rdkit' or 'rdkit-pypi' to requirements.txt."
        )
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        standardized_mol = Chem.MolToSmiles(
            mol,
            isomericSmiles=True,
            canonical=True
        )
        return standardized_mol

    except Exception:
        return None


# Calculate ECFP fingerprint
def calculate_ecfp(smiles, radius=3, bits=1024, chirality=False):
    if Chem is None or AllChem is None:
        st.error(
            "RDKit is not installed. Please add 'rdkit' or 'rdkit-pypi' to requirements.txt."
        )
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius,
            nBits=bits,
            useChirality=chirality
        )

        return np.array(fp, dtype=float)

    except Exception:
        return None


# Add formulation features
def add_formulation_features(ecfp, features_dict):
    features = np.array(list(features_dict.values()), dtype=float)
    return np.concatenate([ecfp, features])


@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Please make sure this file exists in your GitHub repository."
        )

    return joblib.load(model_path)


def predict_with_model(model, features):
    df = pd.DataFrame([features])
    pred = model.predict(df)[0]
    return int(pred)


# Main function
def main():
    st.set_page_config(
        page_title="ILs Property Prediction Platform",
        page_icon="🧪",
        layout="centered"
    )

    st.title("ILs Property Prediction Platform")
    st.markdown("Input SMILES string of ILs for property prediction")
'''
    with st.expander("Deployment dependency note"):
        st.markdown(
            """
            If this app fails on Streamlit Cloud with `ModuleNotFoundError`,
            please make sure your GitHub repository contains a `requirements.txt`
            file including:

            ```txt
            streamlit
            pandas
            numpy
            joblib
            scikit-learn
            lightgbm
            rdkit
            ```

            If `rdkit` fails to install, try replacing it with:

            ```txt
            rdkit-pypi
            ```
            """
        )
 '''
    if Chem is None or AllChem is None:
        st.error(
            "RDKit import failed. Please add `rdkit` or `rdkit-pypi` to `requirements.txt` "
            "and redeploy the app."
        )
        st.stop()

    smiles_input = st.text_input("Enter SMILES string of ILs:", "")

    if st.button("Predict"):
        if not smiles_input.strip():
            st.warning("Please enter a valid SMILES string")
            return

        # Standardize SMILES
        standardized_smiles = standardize_smiles(smiles_input.strip())

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
            "LipRat_IL": 0.5,
            "LipRat_Cho": 0.385,
            "LipRat_Help": 0.1,
            "LipRat_PEG": 0.015,
            "Helper_DPhyPE": 0,
            "Helper_DSPC": 1,
            "PEG_type_DMG-PEG": 1,
            "PEG_type_PEG-DMA": 0,
            "Protein_hEPO": 0,
            "Protein_luciferase": 1,
            "N/P ratio": 6,
            "Mass ratio": -1
        }

        eff_formulation = {
            "LipRat_IL": 0.5,
            "LipRat_Cho": 0.385,
            "LipRat_Help": 0.1,
            "LipRat_PEG": 0.015,
            "Helper_DSPC": 1,
            "PEG_DMG-PEG2000": 1,
            "PEG_DMPE-PEG2000": 0,
            "PEG_PEG-DMA": 0,
            "PEG_PEG2000-C-DMA": 0,
            "Protein_eGEP": 0,
            "Protein_hEPO": 0,
            "Protein_luciferase": 1,
            "Mice_BALB/c": 1,
            "Mice_C57BL/6": 0,
            "Mice_CD-1": 0,
            "Mice_ICR": 0,
            "Time": 4,
            "N/P ratio": 6,
            "Mass ratio": -1
        }

        # Build feature sets
        pka_features = add_formulation_features(ecfp, pka_formulation)
        eff_features = add_formulation_features(ecfp, eff_formulation)
        mem_features = eff_features.copy()

        # Load models
        try:
            pka_model = load_model("pKa_ECFP6_LGB.pkl")
            eff_model = load_model("ecfp6_Efficiency_LGB.pkl")
            mem_model = load_model("LightGBM_model.pkl")

        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            st.error(
                "Please ensure all `.pkl` model files are included in your GitHub repository "
                "and all required packages are listed in `requirements.txt`."
            )
            return

        # Make predictions
        try:
            pka_pred = predict_with_model(pka_model, pka_features)
            eff_pred = predict_with_model(eff_model, eff_features)
            mem_pred = predict_with_model(mem_model, mem_features)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return

        # Interpret prediction results
        pka_result = {
            0: "apparent pKa ≤ 6",
            1: "6 < apparent pKa < 7",
            2: "apparent pKa ≥ 7"
        }.get(pka_pred, f"Unknown prediction class: {pka_pred}")

        eff_result = {
            0: "Deliver mRNA ability < MC3",
            1: "Deliver mRNA ability ≥ MC3"
        }.get(eff_pred, f"Unknown prediction class: {eff_pred}")

        mem_result = {
            0: "Endosomal membrane fusion ability < SM102",
            1: "Endosomal membrane fusion ability ≥ SM102"
        }.get(mem_pred, f"Unknown prediction class: {mem_pred}")

        # Display results
        st.subheader("Prediction Results:")
        st.markdown(f"**Apparent pKa Prediction**: {pka_result}")
        st.markdown(f"**Deliver mRNA Ability Prediction**: {eff_result}")
        st.markdown(f"**Endosomal Escape Ability Prediction**: {mem_result}")


if __name__ == "__main__":
    main()
