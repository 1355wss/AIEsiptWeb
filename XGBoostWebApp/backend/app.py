from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import xgboost as xgb
import numpy as np
import pandas as pd
from smiles_to_features import smiles_to_vector
import io
from rdkit import Chem
from rdkit.Chem import AllChem

app = Flask(__name__)
CORS(app)

model = xgb.XGBRegressor()
model.load_model("AtomCount_XGB_0.59.json")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    smiles_input = data.get("smiles")
    
    # 💡 若是字符串，转换为单元素列表；若是列表，直接用
    if isinstance(smiles_input, str):
        smiles_list = [smiles_input]
    elif isinstance(smiles_input, list):
        smiles_list = smiles_input
    else:
        return jsonify({"error": "input must be a string or list of SMILES"}), 400

    try:
        features_list = [smiles_to_vector(s) for s in smiles_list]  # 批量转换
        predictions = model.predict(features_list)
        return jsonify({"prediction": [float(p) for p in predictions]})  # 统一返回列表格式
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict/download", methods=["POST"])
def predict_and_download():
    data = request.json
    smiles_list = data.get("smiles_list", [])

    if smiles_list and "smiles" in smiles_list[0].lower():
        smiles_list = smiles_list[1:]
    
    records = []

    for smiles in smiles_list:
        try:
            features = smiles_to_vector(smiles)
            pred = float(model.predict([features])[0])
            records.append({"SMILES": smiles, "Prediction": pred})
        except Exception as e:
            records.append({"SMILES": smiles, "Error": str(e)})

    df = pd.DataFrame(records)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.read().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="predictions.csv"
    )

@app.route("/molfile", methods=["POST"])
def molfile():
    data = request.json
    smiles = data.get("smiles")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({"error": "Invalid SMILES"}), 400
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
    mol_block = Chem.MolToMolBlock(mol)
    return jsonify({"molfile": mol_block})

if __name__ == "__main__":
    app.run(debug=True)
