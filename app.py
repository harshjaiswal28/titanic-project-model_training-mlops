import os, logging, joblib, pandas as pd
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
application = app

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_pipeline.pkl")
log.info(f"BASE_DIR={BASE_DIR}  MODEL_PATH={MODEL_PATH}")

_pipeline = None
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        log.info(f"Loading model from {MODEL_PATH}")
        _pipeline = joblib.load(MODEL_PATH)
        log.info("Model loaded OK")
    return _pipeline

FEATURE_COLS = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
DEFAULTS     = {"Pclass":3,"Sex":"male","Age":28.0,"SibSp":0,"Parch":0,"Fare":32.0,"Embarked":"S"}

@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return r

@app.route("/predict", methods=["OPTIONS"])
@app.route("/health",  methods=["OPTIONS"])
def preflight(): return "",204

# ── / returns 200 instantly — critical for Azure health probe ─
@app.route("/", methods=["GET"])
def index():
    return """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Titanic Survival API</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#0f172a;min-height:100vh;
     display:flex;align-items:center;justify-content:center;padding:20px}
.card{background:#fff;border-radius:16px;padding:36px;max-width:540px;width:100%;
      box-shadow:0 25px 60px rgba(0,0,0,.5)}
h1{font-size:1.55rem;color:#0f172a;margin-bottom:2px}
.sub{color:#64748b;font-size:.88rem;margin-bottom:24px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.full{grid-column:1/-1}
label{display:block;font-size:.75rem;font-weight:700;color:#475569;
      text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px}
input,select{width:100%;padding:9px 12px;border:1.5px solid #e2e8f0;
             border-radius:8px;font-size:.95rem}
input:focus,select:focus{outline:none;border-color:#3b82f6}
.btn{width:100%;padding:13px;background:linear-gradient(135deg,#3b82f6,#1d4ed8);
     color:#fff;border:none;border-radius:8px;font-size:1rem;font-weight:600;
     cursor:pointer;margin-top:16px}
.btn:disabled{opacity:.55;cursor:not-allowed}
#res{margin-top:14px;padding:14px;border-radius:8px;font-weight:600;
     font-size:1rem;display:none;text-align:center}
.ok{background:#dcfce7;color:#166534;border:1.5px solid #86efac}
.no{background:#fee2e2;color:#991b1b;border:1.5px solid #fca5a5}
.er{background:#fef9c3;color:#854d0e}
.prob{font-size:.82rem;font-weight:400;margin-top:3px}
.api{margin-top:20px;background:#f8fafc;border-radius:8px;padding:14px;
     font-size:.8rem;color:#475569}
code{background:#e2e8f0;padding:1px 5px;border-radius:4px;word-break:break-all}
</style></head><body>
<div class="card">
  <h1>&#x1F6A2; Titanic Survival Predictor</h1>
  <p class="sub">Azure App Service &middot; ML API</p>
  <div class="grid">
    <div><label>Class</label>
      <select id="Pclass">
        <option value="1">1st</option><option value="2">2nd</option>
        <option value="3" selected>3rd</option></select></div>
    <div><label>Sex</label>
      <select id="Sex">
        <option value="female">Female</option>
        <option value="male" selected>Male</option></select></div>
    <div><label>Age</label>
      <input id="Age" type="number" value="29" min="0" max="100"/></div>
    <div><label>Embarked</label>
      <select id="Embarked">
        <option value="S" selected>S - Southampton</option>
        <option value="C">C - Cherbourg</option>
        <option value="Q">Q - Queenstown</option></select></div>
    <div><label>Siblings/Spouses</label>
      <input id="SibSp" type="number" value="0" min="0"/></div>
    <div><label>Parents/Children</label>
      <input id="Parch" type="number" value="0" min="0"/></div>
    <div class="full"><label>Fare (GBP)</label>
      <input id="Fare" type="number" value="32" min="0" step="0.01"/></div>
  </div>
  <button class="btn" id="btn" onclick="runPredict()">Predict Survival</button>
  <div id="res"></div>
  <div class="api">
    <strong>API Endpoints</strong><br/><br/>
    <code>GET /health</code> &mdash; status check<br/>
    <code>POST /predict</code> &mdash; JSON prediction<br/><br/>
    Body: <code>{"Pclass":1,"Sex":"female","Age":29,"SibSp":0,"Parch":0,"Fare":151,"Embarked":"S"}</code>
  </div>
</div>
<script>
async function runPredict(){
  const btn=document.getElementById('btn'),res=document.getElementById('res');
  btn.disabled=true;btn.textContent='Predicting...';res.style.display='none';
  const p={Pclass:+document.getElementById('Pclass').value,
           Sex:document.getElementById('Sex').value,
           Age:+document.getElementById('Age').value,
           SibSp:+document.getElementById('SibSp').value,
           Parch:+document.getElementById('Parch').value,
           Fare:+document.getElementById('Fare').value,
           Embarked:document.getElementById('Embarked').value};
  try{
    const r=await fetch('/predict',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});
    const d=await r.json();
    if(!r.ok)throw new Error(d.error||'Server error');
    const item=Array.isArray(d)?d[0]:d;
    const pct=(item.survival_probability*100).toFixed(1);
    res.className=item.prediction===1?'ok':'no';
    res.innerHTML=item.prediction===1
      ?'Survived<div class="prob">Probability: '+pct+'%</div>'
      :'Did not survive<div class="prob">Probability: '+pct+'%</div>';
  }catch(e){res.className='er';res.innerHTML='Error: '+e.message;}
  res.style.display='block';
  btn.disabled=false;btn.textContent='Predict Survival';
}
</script>
</body></html>""", 200

@app.route("/health", methods=["GET"])
def health():
    exists = os.path.exists(MODEL_PATH)
    return jsonify({
        "status":       "ok",
        "model_file":   exists,
        "model_loaded": _pipeline is not None,
        "base_dir":     BASE_DIR,
        "model_path":   MODEL_PATH,
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        pipeline = get_pipeline()
    except Exception as e:
        return jsonify({"error": f"Model load failed: {e}"}), 503

    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"error": "Missing JSON body"}), 400

    records = [body] if isinstance(body, dict) else body
    try:
        df = pd.DataFrame(records)
        for col, val in DEFAULTS.items():
            if col not in df.columns:
                df[col] = val
        df = df[FEATURE_COLS].copy()
        for col in ["Pclass","SibSp","Parch"]:
            df[col] = pd.to_numeric(df[col],errors="coerce").fillna(DEFAULTS[col]).astype(int)
        for col in ["Age","Fare"]:
            df[col] = pd.to_numeric(df[col],errors="coerce").fillna(DEFAULTS[col])
        df["Sex"]      = df["Sex"].fillna(DEFAULTS["Sex"]).str.lower().str.strip()
        df["Embarked"] = df["Embarked"].fillna(DEFAULTS["Embarked"]).str.upper().str.strip()
        preds = pipeline.predict(df)
        probs = pipeline.predict_proba(df)[:,1]
        return jsonify([{
            "prediction":           int(preds[i]),
            "survived":             "Yes" if preds[i]==1 else "No",
            "survival_probability": round(float(probs[i]),4),
        } for i in range(len(df))]), 200
    except Exception as e:
        log.exception("Prediction error")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
